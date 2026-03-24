from app.services.ranking import cosine_similarity, rank_vector_candidates, rrf_fuse


def test_cosine_similarity_of_identical_vectors_is_one():
    assert round(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 4) == 1.0


def test_rank_vector_candidates_orders_by_similarity():
    ranked = rank_vector_candidates(
        query_vector=[1.0, 0.0],
        candidates=[
            {"message_id": "a", "embedding": [1.0, 0.0]},
            {"message_id": "b", "embedding": [0.0, 1.0]},
        ],
        limit=2,
    )
    assert ranked[0]["message_id"] == "a"


def test_rrf_fuse_merges_two_rank_lists():
    results = rrf_fuse(
        vector_hits=[{"message_id": "a", "score": 0.9}],
        fulltext_hits=[{"message_id": "b", "score": 8.0}, {"message_id": "a", "score": 7.0}],
        top_k=5,
    )
    assert {item["message_id"] for item in results} == {"a", "b"}
