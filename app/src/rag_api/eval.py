"""Retrieval evaluation against a running rag-api: Recall@k, MRR, and what no-answer questions get
back.

    python -m rag_api.eval --url http://127.0.0.1:8000 --collection cms_content \\
        --questions eval/questions.yaml [--k 1 3 5] [--min-dense-score 0.5] [--json report.json]

The token comes from RAG_EVAL_TOKEN (any token that may read the collection). Only POST /query with
include_answer=false is called, so nothing is spent on an LLM.

Question files (YAML):

    collection: cms_content            # optional, --collection wins
    doc_id_template: "blog:{slug}"     # optional: how `slugs` map to doc_ids
    questions:
      - id: c01 question: What is the difference between Helm and Kustomize? expected:
        ["blog:15-essential-kubernetes-tools"]   # any one counts as a hit type: concept
        # reported per type; "no-answer" is special
      - id: n01 question: How do I run Kafka on Windows? type: no-answer
        # nothing should be retrieved convincingly

moks-seo-agent's app/data/rag_eval.yaml (questions with `slugs`) works as is with
--doc-id-template 'blog:{slug}' (moks-cms indexes a post as "<post collection>:<slug>").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

NO_ANSWER = "no-answer"


@dataclass
class Question:
    id: str
    question: str
    expected: list[str]
    type: str = "other"
    tags: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    question: Question
    ranked_doc_ids: list[str]
    top_dense: float | None

    def rank(self) -> int | None:
        for i, doc_id in enumerate(self.ranked_doc_ids, start=1):
            if doc_id in self.question.expected:
                return i
        return None


def load_questions(
    path: Path, doc_id_template: str | None
) -> tuple[list[Question], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    template = doc_id_template or data.get("doc_id_template")
    out: list[Question] = []
    for raw in data.get("questions") or []:
        expected = [str(x) for x in raw.get("expected") or []]
        if not expected and raw.get("slugs"):
            if not template:
                raise SystemExit(f"{raw.get('id')}: has slugs but no --doc-id-template")
            expected = [template.format(slug=s) for s in raw["slugs"]]
        out.append(
            Question(
                id=str(raw.get("id") or len(out) + 1),
                question=str(raw["question"]),
                expected=expected,
                type=str(raw.get("type") or ("no-answer" if not expected else "other")),
                tags=[str(t) for t in raw.get("tags") or []],
            )
        )
    return out, data


def distinct_docs(sources: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for s in sources:
        doc_id = str(s.get("doc_id") or "")
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def summarize(outcomes: list[Outcome], ks: list[int]) -> dict[str, Any]:
    answerable = [o for o in outcomes if o.question.type != NO_ANSWER and o.question.expected]
    no_answer = [o for o in outcomes if o.question.type == NO_ANSWER]

    def block(items: list[Outcome]) -> dict[str, Any]:
        n = len(items)
        if not n:
            return {"questions": 0}
        ranks = [o.rank() for o in items]
        result: dict[str, Any] = {"questions": n}
        for k in ks:
            result[f"recall@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
        result["mrr"] = round(sum(1.0 / r for r in ranks if r) / n, 4)
        return result

    report: dict[str, Any] = {"all": block(answerable), "by_type": {}, "by_tag": {}}
    for t in sorted({o.question.type for o in answerable}):
        report["by_type"][t] = block([o for o in answerable if o.question.type == t])
    for tag in sorted({tag for o in answerable for tag in o.question.tags}):
        report["by_tag"][tag] = block([o for o in answerable if tag in o.question.tags])
    report["no_answer"] = {
        "questions": len(no_answer),
        "returned_something": sum(1 for o in no_answer if o.ranked_doc_ids),
        "top_dense_scores": [o.top_dense for o in no_answer],
    }
    report["misses"] = [
        {"id": o.question.id, "expected": o.question.expected, "got": o.ranked_doc_ids[: max(ks)]}
        for o in answerable
        if o.rank() is None or o.rank() > max(ks)
    ]
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    questions, data = load_questions(Path(args.questions), args.doc_id_template)
    collection = args.collection or data.get("collection")
    if not collection:
        raise SystemExit("--collection is required (or `collection:` in the file)")
    token = os.environ.get("RAG_EVAL_TOKEN", "")
    if not token:
        raise SystemExit("set RAG_EVAL_TOKEN")
    headers = {"Authorization": f"Bearer {token}", "X-Service-Name": "rag-eval"}
    top_k = max(args.k)
    outcomes: list[Outcome] = []
    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"), headers=headers, timeout=60
    ) as client:
        for q in questions:
            body: dict[str, Any] = {
                "collection": collection,
                "query": q.question,
                # Several chunks of one document are common: ask for more, rank distinct documents.
                "top_k": min(50, top_k * 4),
                "include_answer": False,
            }
            if args.min_dense_score is not None:
                body["min_dense_score"] = args.min_dense_score
            resp = await client.post("/query", json=body)
            resp.raise_for_status()
            sources = resp.json().get("sources") or []
            dense = [s.get("dense_score") for s in sources if s.get("dense_score") is not None]
            outcomes.append(Outcome(q, distinct_docs(sources), max(dense) if dense else None))
    report = summarize(outcomes, sorted(args.k))
    report.update(collection=collection, min_dense_score=args.min_dense_score, url=args.url)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--collection")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-id-template")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--min-dense-score", type=float)
    parser.add_argument("--json", help="also write the report to this file")
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
