from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict

import ir_datasets

from app.services.search_service import get_search_service


def recall_at_k(relevant: set[str], ranked_doc_ids: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_doc_ids[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked_doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(qrels: dict[str, int], ranked_doc_ids: list[str], k: int) -> float:
    def dcg(scores: list[int]) -> float:
        total = 0.0
        for i, rel in enumerate(scores, start=1):
            total += (2**rel - 1) / math.log2(i + 1)
        return total

    predicted = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(predicted) / ideal_dcg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beir/scifact")
    parser.add_argument("--strategy", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=200)
    args = parser.parse_args()

    dataset = ir_datasets.load(args.dataset)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels_by_query[qrel.query_id][qrel.doc_id] = qrel.relevance

    service = get_search_service()
    recalls, mrrs, ndcgs, latencies = [], [], [], []

    for idx, (query_id, qrels) in enumerate(qrels_by_query.items()):
        if args.limit_queries and idx >= args.limit_queries:
            break
        query_text = queries[query_id]
        started = time.perf_counter()
        results, latency_ms = service.search(
            query_text,
            top_k=args.k,
            use_reranker=False,
            strategy=args.strategy,
        )
        latencies.append(latency_ms if latency_ms else (time.perf_counter() - started) * 1000)
        ranked_doc_ids = []
        seen = set()
        for item in results:
            if item.chunk.doc_id not in seen:
                ranked_doc_ids.append(item.chunk.doc_id)
                seen.add(item.chunk.doc_id)
        relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
        recalls.append(recall_at_k(relevant, ranked_doc_ids, args.k))
        mrrs.append(reciprocal_rank(relevant, ranked_doc_ids))
        ndcgs.append(ndcg_at_k(qrels, ranked_doc_ids, 10))

    print(
        {
            "dataset": args.dataset,
            "strategy": args.strategy,
            "queries_evaluated": len(recalls),
            f"Recall@{args.k}": round(statistics.mean(recalls), 4) if recalls else 0.0,
            "MRR": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
            "nDCG@10": round(statistics.mean(ndcgs), 4) if ndcgs else 0.0,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        }
    )


if __name__ == "__main__":
    main()
