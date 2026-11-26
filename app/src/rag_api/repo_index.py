"""Index GitHub repositories' documents into rag-api's `github_repos` collection.

    python -m rag_api.repo_index        (CronJob repo-index, deploy/k8s/repo-index.yaml)

What it indexes (the JSON config in REPO_INDEX_CONFIG, ConfigMap repo-index; README "Indexing the
GitHub repositories"):
- every repository of each org in `orgs` that is not archived, plus `extra_repos` ("owner/name"),
  minus `exclude_repos`; a new repository is picked up by the next run without any change here;
- in each, the files of the default branch that match an `include` glob and no `exclude` glob. A
  glob applies to every repository, or to one when written "owner/name:glob". `**/` matches any
  number of directories (none included), `*` anything but "/".

How: one listing call to the GitHub API per org (per extra repository), then the default branch as
one tarball from codeload.github.com (not counted against the API's rate limit). GITHUB_TOKEN, when
set, is sent to the API only (60 calls an hour without it, shared by the cluster's address).

Each file is one rag-api document, sent one at a time and waited for (an ingest takes as long as
bge-m3 needs; rag-api makes a resend wait for the first one, so nothing is embedded twice):
- doc_id "owner/name/path"; title "name · path"; source and metadata `url` the file on GitHub
  (default branch); metadata `repo`, `path`, `branch`, `blob_sha` (git's id of the file's content);
- text files (`TEXT_EXTENSIONS`) go as JSON text; any other included file (PDF, DOCX, PPTX ...) as
  an upload, which rag-api converts with docling-serve. Nothing in a document changes unless the
  file does, so rag-api answers "unchanged" without embedding.

Afterwards, documents the run no longer found are deleted: files removed or newly excluded in a
repository that was read completely, and every document of a repository that is gone (deleted,
archived, excluded). Nothing is deleted when listing a GitHub org failed, or when a repository's
tarball could not be read.

Exit status 1 when anything failed (the Job shows it), 0 otherwise; rag-api refusing one file (422:
no text, a chunk too long) is logged and counted as rejected, not failed.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import tarfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from rag_api.logs import emit

logger = logging.getLogger("rag_api.repo_index")

COLLECTION = "github_repos"
SERVICE = "repo-indexer"
TEXT_EXTENSIONS = frozenset({".md", ".mdx", ".markdown", ".rst", ".txt", ".adoc"})
GITHUB_API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"


# --- configuration --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    orgs: tuple[str, ...] = ()
    extra_repos: tuple[str, ...] = ()
    exclude_repos: frozenset[str] = frozenset()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    max_file_bytes: int | None = None

    @classmethod
    def from_json(cls, raw: str) -> Config:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("the config must be a JSON object")
        max_bytes = data.get("max_file_bytes")
        if max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes < 1):
            raise ValueError("max_file_bytes must be a positive integer or null (no limit)")
        cfg = cls(
            orgs=tuple(_strings(data, "orgs")),
            extra_repos=tuple(_strings(data, "extra_repos")),
            exclude_repos=frozenset(r.lower() for r in _strings(data, "exclude_repos")),
            include=tuple(_strings(data, "include")),
            exclude=tuple(_strings(data, "exclude")),
            max_file_bytes=max_bytes,
        )
        for repo in cfg.extra_repos:
            if repo.count("/") != 1:
                raise ValueError(f"extra_repos entry {repo!r} is not owner/name")
        if not cfg.orgs and not cfg.extra_repos:
            raise ValueError("the config names no orgs and no extra_repos: nothing to index")
        if not cfg.include:
            raise ValueError("the config has no include globs: nothing to index")
        return cfg


def _strings(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return [v.strip() for v in value]


def glob_regex(pattern: str) -> re.Pattern[str]:
    """A path glob as a regex: `**/` any number of directories (none included), `**` anything,
    `*` anything but "/", `?` one character but "/"."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z", re.IGNORECASE)


@dataclass
class Matcher:
    """Which files of one repository are indexed."""

    include: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]

    @classmethod
    def for_repo(cls, cfg: Config, repo: str) -> Matcher:
        def scoped(globs: Iterable[str]) -> list[re.Pattern[str]]:
            out: list[re.Pattern[str]] = []
            for g in globs:
                scope, sep, rest = g.partition(":")
                if sep and "/" in scope and not scope.startswith(("*", "/")):
                    if scope.lower() == repo.lower():
                        out.append(glob_regex(rest))
                else:
                    out.append(glob_regex(g))
            return out

        return cls(include=scoped(cfg.include), exclude=scoped(cfg.exclude))

    def wants(self, path: str) -> bool:
        return any(p.match(path) for p in self.include) and not any(
            p.match(path) for p in self.exclude
        )


# --- GitHub ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Repo:
    full_name: str  # owner/name
    branch: str

    @property
    def name(self) -> str:
        return self.full_name.split("/", 1)[1]


class GitHubError(Exception):
    pass


class GitHub:
    def __init__(self, client: httpx.Client, token: str | None) -> None:
        self._client = client
        self._api_headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        }

    def _get_json(self, url: str) -> tuple[Any, httpx.Response]:
        try:
            resp = self._client.get(url, headers=self._api_headers)
        except httpx.HTTPError as exc:
            raise GitHubError(f"{url}: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            hint = ""
            if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
                hint = " (API rate limit; set GITHUB_TOKEN)"
            raise GitHubError(f"{url}: HTTP {resp.status_code}{hint}")
        return resp.json(), resp

    def org_repos(self, org: str) -> list[dict[str, Any]]:
        url: str | None = f"{GITHUB_API}/orgs/{quote(org)}/repos?per_page=100&type=all"
        repos: list[dict[str, Any]] = []
        while url:
            page, resp = self._get_json(url)
            if not isinstance(page, list):
                raise GitHubError(f"{url}: not a list of repositories")
            repos.extend(page)
            url = resp.links.get("next", {}).get("url")
        return repos

    def repo(self, full_name: str) -> dict[str, Any]:
        data, _ = self._get_json(f"{GITHUB_API}/repos/{full_name}")
        if not isinstance(data, dict):
            raise GitHubError(f"{full_name}: not a repository")
        return data

    def files(
        self, repo: Repo, wants: Matcher, max_bytes: int | None
    ) -> Iterator[tuple[str, bytes]]:
        """(path, content) of each wanted regular file of the default branch, from one tarball
        streamed from codeload (nothing written to disk)."""
        url = f"{CODELOAD}/{repo.full_name}/tar.gz/{quote(repo.branch, safe='')}"
        try:
            with self._client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise GitHubError(f"{url}: HTTP {resp.status_code}")
                stream = io.BufferedReader(_ByteStream(resp.iter_bytes()))
                with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        _, _, path = member.name.partition("/")  # "<name>-<branch>/<path>"
                        if not path or not wants.wants(path):
                            continue
                        if max_bytes is not None and member.size > max_bytes:
                            emit(
                                logger,
                                "repo_index_file_skipped",
                                repo=repo.full_name,
                                path=path,
                                reason="larger than max_file_bytes",
                                bytes=member.size,
                            )
                            continue
                        handle = tar.extractfile(member)
                        if handle is not None:
                            yield path, handle.read()
        except (httpx.HTTPError, tarfile.TarError, OSError, EOFError) as exc:
            raise GitHubError(f"{url}: {type(exc).__name__}: {exc}") from exc


class _ByteStream(io.RawIOBase):
    """A readable file over an iterator of byte chunks (tarfile's streaming mode needs read())."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(buffer), len(self._buf))
        buffer[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def discover(github: GitHub, cfg: Config) -> list[Repo]:
    """The repositories to index. Raises GitHubError when any listing failed: deleting the documents
    of repositories that only look gone would be wrong."""
    found: dict[str, Repo] = {}
    for org in cfg.orgs:
        for data in github.org_repos(org):
            _add(found, data, cfg)
    for full_name in cfg.extra_repos:
        _add(found, github.repo(full_name), cfg)
    return sorted(found.values(), key=lambda r: r.full_name.lower())


def _add(found: dict[str, Repo], data: dict[str, Any], cfg: Config) -> None:
    full_name = str(data.get("full_name") or "")
    if not full_name or data.get("archived") or data.get("disabled"):
        return
    if full_name.lower() in cfg.exclude_repos:
        return
    found[full_name.lower()] = Repo(
        full_name=full_name, branch=str(data.get("default_branch") or "main")
    )


# --- documents ------------------------------------------------------------------------------------


def blob_sha(content: bytes) -> str:
    """git's object id for a file with this content (what `git hash-object` prints)."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content, usedforsecurity=False).hexdigest()


def doc_id(repo: Repo, path: str) -> str:
    return f"{repo.full_name}/{path}"


def file_url(repo: Repo, path: str) -> str:
    return f"https://github.com/{repo.full_name}/blob/{quote(repo.branch)}/{quote(path)}"


def document(repo: Repo, path: str, content: bytes) -> dict[str, Any]:
    url = file_url(repo, path)
    return {
        "collection": COLLECTION,
        "doc_id": doc_id(repo, path),
        "title": f"{repo.name} · {path}",
        "source": url,
        "metadata": {
            "repo": repo.full_name,
            "path": path,
            "branch": repo.branch,
            "url": url,
            "blob_sha": blob_sha(content),
        },
    }


def is_text(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in TEXT_EXTENSIONS


# --- rag-api --------------------------------------------------------------------------------------


class RagError(Exception):
    def __init__(self, message: str, status: int | None) -> None:
        super().__init__(message)
        self.status = status


class RagApi:
    def __init__(self, client: httpx.Client, base_url: str, token: str) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "X-Service-Name": SERVICE}

    def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = self._client.request(
                method, f"{self._base}{path}", headers=self._headers, **kwargs
            )
        except httpx.HTTPError as exc:
            raise RagError(f"rag-api did not answer: {type(exc).__name__}", None) from exc
        if resp.status_code >= 400:
            detail = ""
            try:
                body = resp.json()
                detail = str(body.get("detail") or body.get("error") or "")[:200]
            except ValueError:
                pass
            raise RagError(f"rag-api answered HTTP {resp.status_code}: {detail}", resp.status_code)
        body = resp.json()
        return body if isinstance(body, dict) else {}

    def ingest_text(self, doc: dict[str, Any], text: str) -> dict[str, Any]:
        return self._call("POST", "/ingest", json={**doc, "text": text})

    def ingest_file(self, doc: dict[str, Any], filename: str, content: bytes) -> dict[str, Any]:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        form = {
            "collection": doc["collection"],
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "source": doc["source"],
            "metadata": json.dumps(doc["metadata"]),
        }
        return self._call("POST", "/ingest", data=form, files={"file": (filename, content, mime)})

    def list_doc_ids(self, filters: dict[str, str] | None = None) -> list[str]:
        body = self._call(
            "POST", "/documents/list", json={"collection": COLLECTION, "filters": filters}
        )
        return [str(d) for d in body.get("doc_ids") or []]

    def delete(self, doc_id_: str) -> None:
        self._call("DELETE", f"/documents/{quote(doc_id_)}", params={"collection": COLLECTION})


# --- the run --------------------------------------------------------------------------------------


@dataclass
class Totals:
    repos: int = 0
    files: int = 0
    indexed: int = 0
    unchanged: int = 0
    rejected: int = 0
    failed: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failed += 1
        self.errors.append(message)


def index_repo(github: GitHub, rag: RagApi, cfg: Config, repo: Repo, totals: Totals) -> None:
    matcher = Matcher.for_repo(cfg, repo.full_name)
    started = time.monotonic()
    counts = {"files": 0, "indexed": 0, "unchanged": 0, "rejected": 0, "failed": 0, "deleted": 0}
    try:
        files = list(github.files(repo, matcher, cfg.max_file_bytes))
    except GitHubError as exc:
        totals.fail(f"{repo.full_name}: {exc}")
        emit(
            logger,
            "repo_index_repo_failed",
            level=logging.WARNING,
            repo=repo.full_name,
            error=str(exc),
        )
        return
    seen: set[str] = set()
    for path, content in files:
        doc = document(repo, path, content)
        counts["files"] += 1
        text = content.decode("utf-8", errors="replace") if is_text(path) else None
        if text is not None and not text.strip():
            counts["rejected"] += 1  # empty: not kept, so a stored older version is deleted below
            continue
        # Kept from here on, even when this send fails: the stored version stays until one succeeds.
        seen.add(doc["doc_id"])
        try:
            if text is not None:
                result = rag.ingest_text(doc, text)
            else:
                result = rag.ingest_file(doc, os.path.basename(path), content)
        except RagError as exc:
            if (
                exc.status is not None
                and 400 <= exc.status < 500
                and exc.status not in (401, 403, 408, 429)
            ):
                counts["rejected"] += 1
                emit(
                    logger,
                    "repo_index_file_rejected",
                    repo=repo.full_name,
                    path=path,
                    error=str(exc),
                )
            else:
                counts["failed"] += 1  # added to totals with the other counts below
                totals.errors.append(f"{doc['doc_id']}: {exc}")
                emit(
                    logger,
                    "repo_index_file_failed",
                    level=logging.WARNING,
                    repo=repo.full_name,
                    path=path,
                    error=str(exc),
                )
            continue
        counts["unchanged" if result.get("unchanged") else "indexed"] += 1

    # Files gone from this repository, or no longer included: it was read completely.
    try:
        stored = rag.list_doc_ids({"repo": repo.full_name})
        for old in stored:
            if old not in seen:
                rag.delete(old)
                counts["deleted"] += 1
    except RagError as exc:
        totals.fail(f"{repo.full_name}: removing old documents: {exc}")

    for key, value in counts.items():
        setattr(totals, key, getattr(totals, key) + value)
    emit(
        logger,
        "repo_index_repo",
        repo=repo.full_name,
        branch=repo.branch,
        duration_s=round(time.monotonic() - started, 1),
        **counts,
    )


def remove_gone_repos(rag: RagApi, repos: list[Repo], totals: Totals) -> None:
    """Delete the documents of repositories no longer indexed (deleted, archived, excluded)."""
    keep = {r.full_name.lower() for r in repos}
    try:
        for old in rag.list_doc_ids():
            owner, _, rest = old.partition("/")
            name = rest.partition("/")[0]
            if f"{owner}/{name}".lower() not in keep:
                rag.delete(old)
                totals.deleted += 1
    except RagError as exc:
        totals.fail(f"removing documents of repositories no longer indexed: {exc}")


def run(github: GitHub, rag: RagApi, cfg: Config) -> Totals:
    totals = Totals()
    started = time.monotonic()
    try:
        repos = discover(github, cfg)
    except GitHubError as exc:
        totals.fail(f"listing repositories: {exc}")
        emit(logger, "repo_index_done", level=logging.ERROR, **_summary(totals, started))
        return totals
    totals.repos = len(repos)
    for repo in repos:
        index_repo(github, rag, cfg, repo, totals)
    remove_gone_repos(rag, repos, totals)
    emit(logger, "repo_index_done", **_summary(totals, started))
    return totals


def _summary(totals: Totals, started: float) -> dict[str, Any]:
    return {
        "repos": totals.repos,
        "files": totals.files,
        "indexed": totals.indexed,
        "unchanged": totals.unchanged,
        "rejected": totals.rejected,
        "failed": totals.failed,
        "deleted": totals.deleted,
        "duration_s": round(time.monotonic() - started, 1),
        "errors": totals.errors[:20],
    }


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
    config_path = os.environ.get("REPO_INDEX_CONFIG", "/etc/repo-index/config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = Config.from_json(f.read())
    except (OSError, ValueError) as exc:
        emit(
            logger,
            "repo_index_config_invalid",
            level=logging.ERROR,
            path=config_path,
            error=str(exc),
        )
        return 2
    token = os.environ.get("RAG_BEARER_TOKEN", "")
    if not token:
        emit(
            logger,
            "repo_index_config_invalid",
            level=logging.ERROR,
            error="RAG_BEARER_TOKEN is not set",
        )
        return 2
    rag_url = os.environ.get("RAG_URL", "http://rag-api.rag.svc.cluster.local:8000")
    # No read timeout towards rag-api: an ingest takes as long as bge-m3 needs, and giving up
    # early only makes the caller send it again (README "Limits and tuning"). GitHub keeps
    # httpx's defaults.
    with (
        httpx.Client(follow_redirects=True) as gh_client,
        httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as rag_client,
    ):
        github = GitHub(gh_client, os.environ.get("GITHUB_TOKEN") or None)
        totals = run(github, RagApi(rag_client, rag_url, token), cfg)
    return 1 if totals.failed else 0


if __name__ == "__main__":
    sys.exit(main())
