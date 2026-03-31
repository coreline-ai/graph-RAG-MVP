"""
Run actual chunking + entity extraction pipeline locally (no DB required).
Uses the real project code: WorkbookParser, IssueChunkingService, GraphBuilder.
Optionally runs BGE-M3 embedding if model is available.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.services.behavior_labeler import BehaviorLabeler
from app.services.chunking import ChunkingService
from app.services.graph_builder import GraphBuilder
from app.services.issue_chunking import IssueChunkingService
from app.services.workbook_parser import WorkbookParser


def main() -> None:
    settings = Settings(
        _env_file=str(PROJECT_ROOT / ".env"),
        _env_file_encoding="utf-8",
    )

    xlsx_path = PROJECT_ROOT / "data" / "model_issue_dataset_10000.xlsx"
    file_bytes = xlsx_path.read_bytes()

    # ── 1. Parse workbook ──
    print("=" * 70)
    print("1. Workbook Parsing")
    print("=" * 70)
    t0 = time.perf_counter()
    parser = WorkbookParser(settings)
    parsed = parser.parse_issue_workbook(file_bytes)
    t_parse = time.perf_counter() - t0
    print(f"  Parsed rows: {len(parsed.rows)} / total: {parsed.total_rows}")
    print(f"  Skipped: {parsed.skipped_rows}, Warnings: {len(parsed.warnings)}")
    print(f"  Time: {t_parse:.2f}s")
    if parsed.warnings[:3]:
        for w in parsed.warnings[:3]:
            print(f"  Warning: {w}")

    # ── 2. Chunking ──
    print(f"\n{'=' * 70}")
    print("2. Issue Chunking (with BehaviorLabeler)")
    print("=" * 70)
    t0 = time.perf_counter()
    labeler = BehaviorLabeler(settings)
    issue_chunking = IssueChunkingService(settings, labeler)
    chunks, summary = issue_chunking.build_chunks(
        parsed.rows,
        document_id="local-test-doc",
        default_access_scopes=["public"],
    )
    t_chunk = time.perf_counter() - t0
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Overview chunks: {summary['overview_chunks']}")
    print(f"  Analysis chunks: {summary['analysis_chunks']}")
    print(f"  Time: {t_chunk:.2f}s")

    # Chunk kind breakdown
    from collections import Counter
    kind_dist = Counter(c.metadata.get("chunk_kind", "unknown") for c in chunks)
    split_dist = Counter(c.metadata.get("split_mode", "unknown") for c in chunks)
    flow_dist = Counter(c.metadata.get("flow_name", "") for c in chunks if c.metadata.get("chunk_kind") == "analysis_flow")

    print(f"\n  chunk_kind distribution:")
    for k, cnt in kind_dist.most_common():
        print(f"    {k}: {cnt}")
    print(f"  split_mode distribution:")
    for k, cnt in split_dist.most_common():
        print(f"    {k}: {cnt}")
    if flow_dist:
        print(f"  flow_name distribution:")
        for k, cnt in flow_dist.most_common():
            print(f"    {k}: {cnt}")

    # Chunk length stats
    lengths = [len(c.chunk_text) for c in chunks]
    print(f"\n  Chunk lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    for kind in ["overview", "analysis_flow", "single"]:
        kind_lens = [len(c.chunk_text) for c in chunks if c.metadata.get("chunk_kind") == kind or (kind == "single" and c.metadata.get("split_mode") == "single")]
        if kind_lens:
            print(f"    {kind}: min={min(kind_lens)}, max={max(kind_lens)}, avg={sum(kind_lens)//len(kind_lens)}")

    # ── 3. Entity extraction ──
    print(f"\n{'=' * 70}")
    print("3. Graph Entity Extraction")
    print("=" * 70)
    t0 = time.perf_counter()
    graph_builder = GraphBuilder()
    graph_rows = graph_builder.build_graph_rows(chunks)
    t_graph = time.perf_counter() - t0
    print(f"  Graph rows: {len(graph_rows)}")
    print(f"  Time: {t_graph:.2f}s")

    entity_counter = Counter()
    entity_type_counter = Counter()
    ents_per_chunk = []
    for row in graph_rows:
        typed_ents = row.get("entities", [])
        ents_per_chunk.append(len(typed_ents))
        for e in typed_ents:
            entity_counter[e["name"]] += 1
            entity_type_counter[e["type"]] += 1

    print(f"\n  Unique entities: {len(entity_counter)}")
    print(f"  Entities per chunk: min={min(ents_per_chunk)}, max={max(ents_per_chunk)}, avg={sum(ents_per_chunk)/len(ents_per_chunk):.1f}")
    print(f"\n  Entity types:")
    for t, cnt in entity_type_counter.most_common():
        print(f"    {t}: {cnt}")
    print(f"\n  Top 20 entities:")
    for name, cnt in entity_counter.most_common(20):
        print(f"    {name:30s}: {cnt}")

    # ── 4. Sample chunks ──
    print(f"\n{'=' * 70}")
    print("4. Sample Chunks")
    print("=" * 70)
    samples = [0, len(chunks) // 3, 2 * len(chunks) // 3, len(chunks) - 1]
    for idx in samples:
        c = chunks[idx]
        print(f"\n  --- Chunk #{idx} ({c.metadata.get('chunk_kind', '?')}, {c.metadata.get('split_mode', '?')}) ---")
        print(f"  Length: {len(c.chunk_text)} chars, Tokens: {c.token_count}")
        print(f"  Entities: {c.metadata.get('entities', [])[:8]}")
        print(f"  Text preview: {c.chunk_text[:200]}...")

    # ── 5. Embedding (if available) ──
    print(f"\n{'=' * 70}")
    print("5. Embedding Test (BGE-M3)")
    print("=" * 70)
    try:
        from app.adapters.embedding_provider import BgeM3EmbeddingProvider
        import asyncio

        provider = BgeM3EmbeddingProvider(settings)
        test_texts = [chunks[0].chunk_text, chunks[len(chunks)//2].chunk_text]
        print(f"  Loading BGE-M3 model ({settings.embedding_model})...")
        t0 = time.perf_counter()
        embeddings = asyncio.run(provider.embed_texts(test_texts))
        t_embed = time.perf_counter() - t0
        print(f"  Model loaded + 2 texts embedded in {t_embed:.2f}s")
        print(f"  Vector dimension: {len(embeddings[0])}")
        print(f"  Sample vector (first 5): {[round(v, 4) for v in embeddings[0][:5]]}")

        # Cosine similarity between two chunks
        import math
        dot = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
        norm0 = math.sqrt(sum(a * a for a in embeddings[0]))
        norm1 = math.sqrt(sum(a * a for a in embeddings[1]))
        cosine = dot / (norm0 * norm1) if norm0 and norm1 else 0
        print(f"  Cosine similarity (chunk #0 vs #{len(chunks)//2}): {cosine:.4f}")

        # Full batch embedding estimate
        total_chunks = len(chunks)
        batch_size = settings.embedding_batch_size
        est_time = t_embed / 2 * total_chunks
        print(f"\n  Full embedding estimate:")
        print(f"    Total chunks: {total_chunks}")
        print(f"    Batch size: {batch_size}")
        print(f"    Estimated time: {est_time/60:.1f} min")
    except Exception as exc:
        print(f"  Embedding not available: {exc}")
        print(f"  (Install sentence-transformers + torch to enable)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Parsed rows:       {len(parsed.rows)}")
    print(f"  Total chunks:      {len(chunks)}")
    print(f"  Unique entities:   {len(entity_counter)}")
    print(f"  Parse time:        {t_parse:.2f}s")
    print(f"  Chunk time:        {t_chunk:.2f}s")
    print(f"  Graph build time:  {t_graph:.2f}s")


if __name__ == "__main__":
    main()
