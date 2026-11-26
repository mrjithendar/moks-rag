"""rag_api.repo_index against a fake GitHub (org listing + codeload tarballs) and a fake rag-api."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
import pytest

from rag_api.repo_index import (
    Config,
    GitHub,
    Matcher,
    RagApi,
    blob_sha,
    glob_regex,
    main,
    run,
)

CONFIG = {
    "orgs": ["mrjithendar"],
    "exclude_repos": ["mrjithendar/private-notes"],
    "include": ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.pdf"],
    "exclude": ["**/node_modules/**", "mrjithendar/jithendar.com:content/**"],
}


def tarball(prefix: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        top = tarfile.TarInfo(prefix)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
        for path, data in files.items():
            info = tarfile.TarInfo(f"{prefix}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{prefix}/LINK.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "README.md"
        tar.addfile(link)
    return buf.getvalue()


REPOS = [
    {"full_name": "mrjithendar/moks-rag", "default_branch": "main", "archived": False},
    {"full_name": "mrjithendar/jithendar.com", "default_branch": "master", "archived": False},
    {"full_name": "mrjithendar/newsroom", "default_branch": "master", "archived": True},
    {"full_name": "mrjithendar/private-notes", "default_branch": "main", "archived": False},
]
TARBALLS = {
    "mrjithendar/moks-rag/main": {
        "README.md": b"# moks-rag\n\nThe shared hybrid RAG platform.\n",
        "docs/guide.md": b"# Guide\n\nHow to ingest.\n",
        "docs/spec.pdf": b"%PDF-1.7 fake",
        "app/main.py": b"print('not a document')\n",
        "empty.md": b"   \n",
        "ui/node_modules/pkg/README.md": b"# a dependency\n",
    },
    "mrjithendar/jithendar.com/master": {
        "content/blog/post.mdx": b"# A blog post\n\nIndexed by moks-cms instead.\n",
        "docs/knowledge.md": b"# Knowledge\n\nShared notes.\n",
    },
}


class FakeGitHub:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.fail_listing = False
        self.fail_tarball: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if request.url.host == "api.github.com":
            if self.fail_listing:
                return httpx.Response(500)
            assert request.url.path == "/orgs/mrjithendar/repos"
            return httpx.Response(200, json=REPOS)
        assert request.url.host == "codeload.github.com"
        owner, name, _, branch = request.url.path.strip("/").split("/", 3)
        key = f"{owner}/{name}/{unquote(branch)}"
        if key in self.fail_tarball:
            return httpx.Response(200, content=b"not a tarball")
        return httpx.Response(200, content=tarball(f"{name}-{branch}", TARBALLS[key]))


class FakeRag:
    """rag-api's /ingest, /documents/list and DELETE /documents/{doc_id} over a dict."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.ingests: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.answer: dict[str, int] = {}  # doc_id -> status to answer instead

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer indexer-token"
        assert request.headers["x-service-name"] == "repo-indexer"
        path = urlsplit(str(request.url)).path
        if request.method == "POST" and path == "/ingest":
            if request.headers["content-type"].startswith("multipart/form-data"):
                body = request.content.decode("latin-1")
                fields = {
                    name: body.split(f'name="{name}"\r\n\r\n', 1)[1].split("\r\n--", 1)[0]
                    for name in ("collection", "doc_id", "title", "source", "metadata")
                }
                doc = {**fields, "metadata": json.loads(fields["metadata"]), "upload": True}
            else:
                doc = {**json.loads(request.content), "upload": False}
            self.ingests.append(doc)
            status = self.answer.get(doc["doc_id"])
            if status:
                return httpx.Response(status, json={"detail": "refused"})
            unchanged = self.docs.get(doc["doc_id"]) == doc["metadata"]
            self.docs[doc["doc_id"]] = doc["metadata"]
            return httpx.Response(200, json={"doc_id": doc["doc_id"], "unchanged": unchanged})
        if request.method == "POST" and path == "/documents/list":
            body = json.loads(request.content)
            assert body["collection"] == "github_repos"
            filters = body.get("filters") or {}
            ids = [
                d
                for d, meta in self.docs.items()
                if all(meta.get(k) == v for k, v in filters.items())
            ]
            return httpx.Response(200, json={"collection": "github_repos", "doc_ids": sorted(ids)})
        if request.method == "DELETE" and path.startswith("/documents/"):
            assert request.url.params["collection"] == "github_repos"
            doc_id = unquote(path.removeprefix("/documents/"))
            self.docs.pop(doc_id, None)
            self.deleted.append(doc_id)
            return httpx.Response(200, json={"doc_id": doc_id, "deleted": True})
        return httpx.Response(404)


def clients(gh: FakeGitHub, rag: FakeRag) -> tuple[GitHub, RagApi]:
    github = GitHub(httpx.Client(transport=httpx.MockTransport(gh.handler)), None)
    api = RagApi(
        httpx.Client(transport=httpx.MockTransport(rag.handler)), "http://rag", "indexer-token"
    )
    return github, api


def stored(repo: str, path: str) -> dict[str, Any]:
    return {"repo": repo, "path": path}


# --- matching -------------------------------------------------------------------------------------


def test_globs_match_like_git_paths() -> None:
    md = glob_regex("**/*.md")
    assert md.match("README.md") and md.match("docs/a/b.md") and md.match("Docs/X.MD")
    assert not md.match("README.mdx")
    root_only = glob_regex("*.md")
    assert root_only.match("README.md") and not root_only.match("docs/a.md")
    assert glob_regex("docs/**").match("docs/a/b.txt") and not glob_regex("docs/**").match(
        "x/docs/a"
    )
    assert glob_regex("**/node_modules/**").match("ui/node_modules/pkg/README.md")


def test_a_repo_scoped_glob_applies_to_that_repo_only() -> None:
    cfg = Config.from_json(json.dumps(CONFIG))
    site = Matcher.for_repo(cfg, "mrjithendar/jithendar.com")
    other = Matcher.for_repo(cfg, "mrjithendar/moks-rag")
    assert not site.wants("content/blog/post.mdx") and site.wants("docs/knowledge.md")
    assert other.wants("content/blog/post.mdx")
    assert not other.wants("app/main.py") and not other.wants("ui/node_modules/x/README.md")


@pytest.mark.parametrize(
    "bad",
    [
        {"include": ["**/*.md"]},  # nothing to index
        {"orgs": ["mrjithendar"]},  # no include globs
        {"orgs": ["mrjithendar"], "include": ["*.md"], "extra_repos": ["no-owner"]},
        {"orgs": "mrjithendar", "include": ["*.md"]},
        {"orgs": ["mrjithendar"], "include": ["*.md"], "max_file_bytes": 0},
    ],
)
def test_bad_config_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Config.from_json(json.dumps(bad))


def test_blob_sha_is_gits() -> None:
    assert blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


# --- a run ----------------------------------------------------------------------------------------


def test_a_run_indexes_wanted_files_and_removes_what_is_gone() -> None:
    gh, rag = FakeGitHub(), FakeRag()
    rag.docs = {
        "mrjithendar/moks-rag/old.md": stored("mrjithendar/moks-rag", "old.md"),  # file deleted
        "mrjithendar/jithendar.com/content/blog/post.mdx": stored(
            "mrjithendar/jithendar.com", "content/blog/post.mdx"
        ),  # now excluded
        "mrjithendar/newsroom/README.md": stored("mrjithendar/newsroom", "README.md"),  # archived
    }
    totals = run(*clients(gh, rag), Config.from_json(json.dumps(CONFIG)))

    assert {d["doc_id"] for d in rag.ingests} == {
        "mrjithendar/moks-rag/README.md",
        "mrjithendar/moks-rag/docs/guide.md",
        "mrjithendar/moks-rag/docs/spec.pdf",
        "mrjithendar/jithendar.com/docs/knowledge.md",
    }
    readme = next(d for d in rag.ingests if d["doc_id"] == "mrjithendar/moks-rag/README.md")
    assert readme["upload"] is False and readme["collection"] == "github_repos"
    assert readme["text"].startswith("# moks-rag")
    assert readme["title"] == "moks-rag · README.md"
    url = "https://github.com/mrjithendar/moks-rag/blob/main/README.md"
    assert readme["source"] == url
    content = TARBALLS["mrjithendar/moks-rag/main"]["README.md"]
    assert readme["metadata"] == {
        "repo": "mrjithendar/moks-rag",
        "path": "README.md",
        "branch": "main",
        "url": url,
        "blob_sha": hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest(),
    }
    pdf = next(d for d in rag.ingests if d["doc_id"].endswith(".pdf"))
    assert pdf["upload"] is True and pdf["metadata"]["path"] == "docs/spec.pdf"

    assert sorted(rag.deleted) == [
        "mrjithendar/jithendar.com/content/blog/post.mdx",
        "mrjithendar/moks-rag/old.md",
        "mrjithendar/newsroom/README.md",
    ]
    # The archived and the excluded repository are never downloaded.
    assert not any("newsroom" in u or "private-notes" in u for u in gh.requests if "codeload" in u)
    assert (totals.repos, totals.files, totals.indexed, totals.rejected) == (2, 5, 4, 1)
    assert (totals.failed, totals.deleted) == (0, 3)

    again = run(*clients(gh, rag), Config.from_json(json.dumps(CONFIG)))
    assert (again.indexed, again.unchanged, again.deleted) == (0, 4, 0)


def test_nothing_is_deleted_when_the_org_listing_fails() -> None:
    gh, rag = FakeGitHub(), FakeRag()
    gh.fail_listing = True
    rag.docs = {"mrjithendar/moks-rag/README.md": stored("mrjithendar/moks-rag", "README.md")}
    totals = run(*clients(gh, rag), Config.from_json(json.dumps(CONFIG)))
    assert totals.failed == 1 and "listing repositories" in totals.errors[0]
    assert rag.deleted == [] and rag.ingests == []


def test_an_unreadable_tarball_keeps_that_repos_documents() -> None:
    gh, rag = FakeGitHub(), FakeRag()
    gh.fail_tarball = {"mrjithendar/moks-rag/main"}
    rag.docs = {"mrjithendar/moks-rag/README.md": stored("mrjithendar/moks-rag", "README.md")}
    totals = run(*clients(gh, rag), Config.from_json(json.dumps(CONFIG)))
    assert totals.failed == 1 and "mrjithendar/moks-rag" in totals.errors[0]
    assert "mrjithendar/moks-rag/README.md" in rag.docs and rag.deleted == []
    assert [d["doc_id"] for d in rag.ingests] == ["mrjithendar/jithendar.com/docs/knowledge.md"]


def test_a_refused_file_is_rejected_and_a_failed_one_is_kept() -> None:
    gh, rag = FakeGitHub(), FakeRag()
    rag.docs = {
        "mrjithendar/moks-rag/docs/guide.md": stored("mrjithendar/moks-rag", "docs/guide.md")
    }
    rag.answer = {
        "mrjithendar/moks-rag/docs/spec.pdf": 422,  # e.g. docling found no text
        "mrjithendar/moks-rag/docs/guide.md": 504,  # rag-api's dependency too slow
    }
    totals = run(*clients(gh, rag), Config.from_json(json.dumps(CONFIG)))
    assert totals.rejected == 2  # the 422 and empty.md
    assert totals.failed == 1 and "docs/guide.md" in totals.errors[0]
    # The stored guide stays (the next run sends it again); nothing of moks-rag is deleted.
    assert "mrjithendar/moks-rag/docs/guide.md" in rag.docs
    assert not any(d.startswith("mrjithendar/moks-rag/") for d in rag.deleted)


def test_files_over_max_file_bytes_are_skipped() -> None:
    gh, rag = FakeGitHub(), FakeRag()
    cfg = Config.from_json(json.dumps({**CONFIG, "max_file_bytes": 20}))
    run(*clients(gh, rag), cfg)
    assert "mrjithendar/moks-rag/README.md" not in {d["doc_id"] for d in rag.ingests}
    assert "mrjithendar/moks-rag/docs/spec.pdf" in {d["doc_id"] for d in rag.ingests}


def test_main_refuses_a_missing_token_or_config(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("REPO_INDEX_CONFIG", str(config))
    monkeypatch.delenv("RAG_BEARER_TOKEN", raising=False)
    assert main() == 2
    monkeypatch.setenv("RAG_BEARER_TOKEN", "t")
    monkeypatch.setenv("REPO_INDEX_CONFIG", str(tmp_path / "missing.json"))
    assert main() == 2
