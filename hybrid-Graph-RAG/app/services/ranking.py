from collections import defaultdict

import numpy as np


def cosine_similarity(query_vector: list[float], candidate_vector: list[float]) -> float:
    query = np.array(query_vector, dtype=float)
    candidate = np.array(candidate_vector, dtype=float)
    denominator = np.linalg.norm(query) * np.linalg.norm(candidate)
    if denominator == 0:
        return 0.0
    return float(np.dot(query, candidate) / denominator)


def rank_vector_candidates(
    query_vector: list[float], candidates: list[dict], limit: int
) -> list[dict]:
    scored = []
    for candidate in candidates:
        embedding = candidate.get("embedding") or []
        score = cosine_similarity(query_vector, embedding)
        item = dict(candidate)
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def rrf_fuse(
    vector_hits: list[dict],
    fulltext_hits: list[dict],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict]:
    merged: dict[str, dict] = {}
    contributions: dict[str, dict] = defaultdict(dict)

    for rank, hit in enumerate(vector_hits, start=1):
        message_id = hit["message_id"]
        merged.setdefault(message_id, dict(hit))
        contributions[message_id]["vector"] = hit["score"]
        contributions[message_id]["rrf"] = contributions[message_id].get("rrf", 0.0) + (
            1.0 / (rrf_k + rank)
        )

    for rank, hit in enumerate(fulltext_hits, start=1):
        message_id = hit["message_id"]
        merged.setdefault(message_id, dict(hit))
        contributions[message_id]["fulltext"] = hit["score"]
        contributions[message_id]["rrf"] = contributions[message_id].get("rrf", 0.0) + (
            1.0 / (rrf_k + rank)
        )

    results = []
    for message_id, payload in merged.items():
        payload["scores"] = {
            "vector": contributions[message_id].get("vector"),
            "fulltext": contributions[message_id].get("fulltext"),
            "rrf": contributions[message_id].get("rrf", 0.0),
        }
        results.append(payload)

    results.sort(key=lambda item: item["scores"]["rrf"], reverse=True)
    return results[:top_k]
