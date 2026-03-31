# CLAUDE.md - Simple-Graph-RAG Project Rules

## Architecture Overview

- **Stack**: FastAPI + PostgreSQL(pgvector) + Neo4j + BGE-M3 + KSS
- **Data**: 10,000 issue records → ~29,762 chunks (overview + analysis flow)
- **Search**: Hybrid vector(pgvector cosine) + graph(Neo4j entity) + metadata filter

## Critical Rules (Past Mistakes — Do NOT Repeat)

### 1. NEVER use `search_text` for query embedding

- **File**: `retrieval.py:365`, `strategies/mixed_issue_chat.py:51`
- **Rule**: Always use `analysis.clean_question` for BGE-M3 embedding
- **Why**: `search_text` is kiwi-tokenized keywords. kiwi splits compound terms:
  - `"타임아웃"` → `"타임" + "아웃"` (meaning destroyed)
  - `"BGE-M3-Retrieve"` → `"BGE" + "M" + "3" + "Retrieve"` (system name destroyed)
- **Result**: Vector search returns irrelevant results because the query vector occupies a different embedding space than the stored chunk vectors
- `search_text` is still used for entity matching / lexical scoring — that's fine

### 2. NEVER create HNSW index BEFORE bulk insert

- **Rule**: Insert all data first, then CREATE INDEX
- **Why**: HNSW index rebuilds on every INSERT. 29,762 rows with index = 13+ hours stuck. Without index = 74 seconds + 22 seconds for index creation = 97 seconds total
- **Pattern**:
  ```sql
  -- 1. Insert all chunks (no HNSW index)
  INSERT INTO chunk_embeddings ...
  -- 2. Create HNSW index AFTER data
  CREATE INDEX ce_hnsw_idx ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
  ```

### 3. NEVER raise `excel_row_max_chars` to eliminate splitting

- **File**: `config.py:38` — currently `600`
- **Rule**: Keep at 600. Do NOT raise to 800+ to avoid splitting
- **Why**: The analysis text (문제점 분석 내용) contains 5 semantic sections (원인/근거/판단/영향/조치). KSS-based splitting via BehaviorLabeler creates separate flow chunks for fine-grained semantic search. If everything becomes a single chunk, searching for "원인 분석" returns the entire 600-char block instead of the precise cause section
- **Correct split**: overview (metadata) + flow 1 (원인 발견) + flow 2 (영향 및 조치) = 3 chunks per issue

### 4. Entity stopwords MUST be synchronized between graph_builder and query_analyzer

- **File**: `app/services/entity_stopwords.py` (shared module)
- **Rule**: Add/remove stopwords in `entity_stopwords.py` ONLY — never directly in `graph_builder.py` or `query_analyzer.py`
- **Why**: If a word is filtered from chunk entities but not from query entities (or vice versa), graph-based entity matching breaks completely — one side has the node, the other doesn't
- **Currently filtered**: structural tags (등록일, 담당자, 진행...) + generic verbs (설정, 로그, 결과, 패치)
- **Intentionally NOT filtered**: `"확인"` — removing it zeros out graph reach for "확인 결과" queries

### 5. Status values MUST be mapped in TWO places

- **Files**: `korean_nlp.py:_STATUS_MAP` AND `query_analyzer.py:_STATUS_KEYWORDS`
- **Rule**: When adding a new status value, update BOTH
- **Why**: `_STATUS_MAP` normalizes the value, `_STATUS_KEYWORDS` tells the query analyzer to recognize it as a status. If only one is updated, the status filter silently fails
- **Current statuses**: 완료, 진행중, 검증중, 분석중, 대기

### 6. BehaviorLabeler merge groups must NOT cross semantic boundaries

- **File**: `behavior_labeler.py:_FLOW_NAME_MAP`
- **Rule**: Do NOT add `frozenset({"discovery", "result", "next_action"})` — this merges ALL 5 sections into 1 chunk, destroying semantic search granularity
- **Correct groups**:
  - Cause group: `discovery` (원인 요약 + 확인 근거 + 기술 판단)
  - Impact group: `result` + `next_action` (영향 범위 + 추가 조치)
- These two groups must stay as **separate chunks**

### 7. CPU-only embedding takes ~15 hours for 29k chunks

- **Config**: `embedding_device: str = "cpu"` in config.py
- On Mac M4: change to `"mps"` → ~12-15 minutes
- On NVIDIA GPU: change to `"cuda"` → ~5-6 minutes
- `embedding_cache` (PostgreSQL table) caches results — second run is seconds
- Always check device before running full ingest

## Project Structure (Key Files)

```
app/
  config.py                    — Settings (excel_row_max_chars=600, embedding_model, etc.)
  services/
    entity_stopwords.py        — SHARED stopwords (graph + query synchronized)
    behavior_labeler.py        — KSS sentence split + flow merge (_FLOW_NAME_MAP)
    issue_chunking.py          — Issue row → overview + analysis flow chunks
    retrieval.py:365           — Query embedding (MUST use clean_question)
    query_analyzer.py          — Query parse + entity extract + status filter
    korean_nlp.py              — Status normalization map
    ranking_policy.py          — Score weights (vector/graph/entity/metadata/recency)
    graph_builder.py           — Entity extraction (uses GRAPH_ENTITY_STOPWORDS)
  adapters/
    embedding_provider.py      — BGE-M3 encode (normalize_embeddings=True)
    postgres_vector_store.py   — pgvector HNSW cosine search
    embedding_cache_store.py   — SHA256 key cache for embeddings
scripts/
  generate_unique_dataset.py   — 10,000 unique issue generator (seed=42)
  ingest_file.py               — CLI ingest (--skip-graph for PG-only)
data/
  model_issue_dataset_10000.xlsx — 10,000 unique issues, 5-section analysis
```

## Docker

```bash
docker compose up -d          # PostgreSQL:5432 + Neo4j:8768
docker compose ps             # Wait for both healthy
```

## Ingest Command

```bash
# Full (PG + Neo4j)
.venv/Scripts/python.exe scripts/ingest_file.py data/model_issue_dataset_10000.xlsx --replace-filename

# PG-only (skip Neo4j graph writes)
.venv/Scripts/python.exe scripts/ingest_file.py data/model_issue_dataset_10000.xlsx --replace-filename --skip-graph
```

## Current Limitations

- **No LLM proxy**: Codex proxy (port 8800) is not configured. Search returns raw sources, no summarized answer
- **Neo4j graph data**: Not populated yet (ingest used --skip-graph equivalent). Graph entity seed search works but graph expansion is limited
- **CPU embedding**: First query takes ~68s (model load). Subsequent queries ~4s
