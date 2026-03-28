from __future__ import annotations

import sys
import types
from datetime import date, time

if "psycopg_pool" not in sys.modules:
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = object
    sys.modules["psycopg_pool"] = psycopg_pool

from app.adapters.codex_proxy import CodexProxyError
from app.config import Settings
from app.schemas import GraphExpansion, QueryFilters, RetrievedChunk
from app.services.query_analyzer import QueryAnalyzer
from app.services.retrieval import RetrievalService


def make_chunk(
    chunk_id: str,
    *,
    document_type: str = "chat",
    vector_score: float = 0.8,
    channel: str = "general",
    user_name: str = "민수",
    message_date: date | None = None,
    chunk_text: str | None = None,
    metadata: dict | None = None,
) -> RetrievedChunk:
    default_metadata = metadata or {"entities": [], "original_lines": [f"[테스트 라인 {chunk_id}]"]}
    if document_type == "issue":
        default_metadata.setdefault("chunk_kind", "overview")
        default_metadata.setdefault("issue_title", chunk_text or chunk_id)
        default_metadata.setdefault("assignee", user_name)
        default_metadata.setdefault("status", "진행")
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_type=document_type,
        channel=channel,
        user_name=user_name,
        message_date=message_date or date(2026, 3, 20),
        message_time=time(14, 30),
        access_scopes=["public"],
        chunk_text=chunk_text or f"{channel} {user_name}: 테스트 청크 {chunk_id}",
        metadata=default_metadata,
        vector_score=vector_score,
    )


class FakePostgres:
    def __init__(self, seed_chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        self.last_search_filters: QueryFilters | None = None
        self.set_chunks(seed_chunks or [])

    def set_chunks(self, chunks: list[RetrievedChunk]) -> None:
        for chunk in chunks:
            self._chunks_by_id[chunk.chunk_id] = chunk

    def _all_chunks(self) -> list[RetrievedChunk]:
        return list(self._chunks_by_id.values())

    def _matches(self, chunk: RetrievedChunk, filters: QueryFilters) -> bool:
        if filters.all_document_types and chunk.document_type not in filters.all_document_types:
            return False
        if filters.all_channels and chunk.channel not in filters.all_channels:
            return False
        if filters.user_names and chunk.user_name not in filters.user_names:
            return False
        if filters.assignees:
            assignee = chunk.metadata.get("assignee") or chunk.user_name
            if assignee not in filters.assignees:
                return False
        if filters.statuses and chunk.metadata.get("status") not in filters.statuses:
            return False
        if filters.access_scopes and not (set(chunk.access_scopes) & set(filters.access_scopes)):
            return False
        if filters.date_from and chunk.message_date < filters.date_from:
            return False
        if filters.date_to and chunk.message_date > filters.date_to:
            return False
        return True

    async def list_channels(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        chunks = self._all_chunks()
        if document_type in {"chat", "issue"}:
            chunks = [chunk for chunk in chunks if chunk.document_type == document_type]
        return sorted({chunk.channel for chunk in chunks})[:limit]

    async def list_users(self, limit: int = 200, document_type: str | None = None) -> list[str]:
        chunks = self._all_chunks()
        if document_type in {"chat", "issue"}:
            chunks = [chunk for chunk in chunks if chunk.document_type == document_type]
        return sorted({chunk.user_name for chunk in chunks})[:limit]

    async def list_assignees(self, document_type: str = "issue", limit: int = 200) -> list[str]:
        chunks = [chunk for chunk in self._all_chunks() if chunk.document_type == "issue"]
        return sorted({
            str(chunk.metadata.get("assignee") or chunk.user_name)
            for chunk in chunks
            if str(chunk.metadata.get("assignee") or chunk.user_name).strip()
        })[:limit]

    async def list_statuses(self, document_type: str = "issue", limit: int = 200) -> list[str]:
        chunks = [chunk for chunk in self._all_chunks() if chunk.document_type == "issue"]
        return sorted({
            str(chunk.metadata.get("status") or "").strip()
            for chunk in chunks
            if str(chunk.metadata.get("status") or "").strip()
        })[:limit]

    async def list_document_types(self, limit: int = 200) -> list[str]:
        return sorted({chunk.document_type for chunk in self._all_chunks()})[:limit]

    async def get_latest_event_date(self, document_type: str | None = None) -> date | None:
        chunks = self._all_chunks()
        if document_type in {"chat", "issue"}:
            chunks = [chunk for chunk in chunks if chunk.document_type == document_type]
        if not chunks:
            return None
        return max(chunk.message_date for chunk in chunks)

    async def search_chunks(self, query_embedding, filters, top_k) -> list[RetrievedChunk]:
        del query_embedding
        self.last_search_filters = filters
        matched = [chunk for chunk in self._all_chunks() if self._matches(chunk, filters)]
        matched.sort(key=lambda chunk: chunk.vector_score, reverse=True)
        return matched[:top_k]

    async def search_issue_candidates(self, query_embedding, filters, top_k) -> list[RetrievedChunk]:
        return await self.search_chunks(query_embedding, filters, top_k)

    async def search_chat_candidates(self, query_embedding, filters, top_k) -> list[RetrievedChunk]:
        return await self.search_chunks(query_embedding, filters, top_k)

    async def get_chunks_by_ids(self, chunk_ids) -> list[RetrievedChunk]:
        ids = set(chunk_ids)
        return [chunk for chunk in self._all_chunks() if chunk.chunk_id in ids]

    async def summarize_filtered_results(self, filters, *, limit: int) -> dict:
        matched = [
            chunk
            for chunk in self._all_chunks()
            if self._matches(chunk, filters)
            and not (chunk.document_type == "issue" and chunk.metadata.get("chunk_kind") != "overview")
        ]
        matched.sort(key=lambda chunk: chunk.message_date, reverse=True)
        return {
            "matched_count": len(matched),
            "count_basis": "matching_records",
            "sample_chunks": matched[:limit],
        }

    async def summarize_special_keyword_results(
        self,
        *,
        filters,
        alias_groups: list[tuple[str, ...]],
        exact_groups: list[tuple[str, ...]],
        limit: int,
    ) -> dict:
        base_matches = [
            chunk
            for chunk in self._all_chunks()
            if self._matches(chunk, filters)
            and chunk.document_type == "issue"
            and chunk.metadata.get("chunk_kind") == "overview"
        ]

        def text_of(chunk: RetrievedChunk) -> str:
            title = str(chunk.metadata.get("issue_title") or "")
            return f"{chunk.chunk_text} {title}".lower()

        def filter_groups(groups: list[tuple[str, ...]]) -> list[RetrievedChunk]:
            if not groups:
                return []
            return [
                chunk
                for chunk in base_matches
                if any(alias in text_of(chunk) for group in groups for alias in group)
            ]

        matched = filter_groups(exact_groups)
        count_basis = "special_exact_records"
        if not matched:
            matched = filter_groups(alias_groups)
            count_basis = "special_alias_records"
        matched.sort(key=lambda chunk: chunk.message_date, reverse=True)
        return {
            "matched_count": len(matched),
            "count_basis": count_basis,
            "sample_chunks": matched[:limit],
        }


class FakeNeo4j:
    def __init__(
        self,
        entity_chunk_ids: list[str] | None = None,
        expansions: dict[str, GraphExpansion] | None = None,
    ) -> None:
        self._entity_chunk_ids = entity_chunk_ids or []
        self._expansions = expansions or {}

    async def find_chunks_by_entities(self, entity_names, *, limit=20) -> list[str]:
        del entity_names
        return self._entity_chunk_ids[:limit]

    async def expand_from_seed_chunks(self, chunk_ids, *, next_window=2) -> dict[str, GraphExpansion]:
        del next_window
        return {chunk_id: expansion for chunk_id, expansion in self._expansions.items() if chunk_id in chunk_ids}

    async def expand_via_entity_cooccurrence(self, chunk_ids, *, entity_names, limit=20) -> list[dict]:
        del chunk_ids, entity_names, limit
        return []

    async def expand_via_same_author(self, chunk_ids, *, limit=10) -> list[dict]:
        del chunk_ids, limit
        return []

    async def extract_subgraph(self, chunk_ids) -> list[dict]:
        del chunk_ids
        return []

    async def find_communities_for_entities(self, entity_names) -> list[dict]:
        del entity_names
        return []

    async def get_entity_cooccurrence_network(self, entity_names, *, limit=50) -> list[dict]:
        del entity_names, limit
        return []

    async def get_entity_mention_counts(self, entity_names) -> dict[str, int]:
        return {str(name): 1 for name in entity_names}


class FakeEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class FixedCodexProxy:
    def __init__(self, response_text: str = "LLM 응답입니다.") -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    async def generate(self, *, system_prompt, user_prompt, metadata=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata or {},
            }
        )

        class _Response:
            text = self.response_text

        return _Response()


class EchoCodexProxy(FixedCodexProxy):
    async def generate(self, *, system_prompt, user_prompt, metadata=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata or {},
            }
        )

        class _Response:
            text = user_prompt

        return _Response()


class FailingCodexProxy:
    async def generate(self, *, system_prompt, user_prompt, metadata=None):
        del system_prompt, user_prompt, metadata
        raise CodexProxyError("timeout", "llm_unavailable")


def build_service(
    *,
    chunks: list[RetrievedChunk],
    entity_chunk_ids: list[str] | None = None,
    expansions: dict[str, GraphExpansion] | None = None,
    codex_proxy=None,
    settings: Settings | None = None,
) -> RetrievalService:
    return RetrievalService(
        settings=settings or Settings(),
        postgres=FakePostgres(chunks),
        neo4j=FakeNeo4j(entity_chunk_ids, expansions),
        embedding_provider=FakeEmbeddingProvider(),
        codex_proxy=codex_proxy or FixedCodexProxy(),
        query_analyzer=QueryAnalyzer(),
    )


def sample_corpus() -> list[RetrievedChunk]:
    return [
        make_chunk(
            "issue-timeout-overview",
            document_type="issue",
            vector_score=0.88,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 18),
            chunk_text="[이슈] API gateway timeout 증가",
            metadata={
                "entities": ["timeout", "gateway"],
                "chunk_kind": "overview",
                "issue_title": "API gateway timeout 증가",
                "assignee": "Sujin",
                "status": "진행",
            },
        ),
        make_chunk(
            "issue-timeout-flow",
            document_type="issue",
            vector_score=0.73,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 18),
            chunk_text="[이슈] API gateway timeout 증가\n[수정 및 결과] 재시도 정책을 조정했다",
            metadata={
                "entities": ["timeout", "gateway"],
                "chunk_kind": "analysis_flow",
                "issue_title": "API gateway timeout 증가",
                "assignee": "Sujin",
                "status": "진행",
                "flow_name": "수정 및 결과",
            },
        ),
        make_chunk(
            "issue-gpu-overview",
            document_type="issue",
            vector_score=0.91,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 19),
            chunk_text="[이슈] 장문 요청 처리 중 GPU 메모리 부족 발생",
            metadata={
                "entities": ["GPU", "메모리 부족", "OOM"],
                "chunk_kind": "overview",
                "issue_title": "장문 요청 처리 중 GPU 메모리 부족 발생",
                "assignee": "Sujin",
                "status": "완료",
            },
        ),
        make_chunk(
            "issue-gpu-flow",
            document_type="issue",
            vector_score=0.82,
            channel="이슈데이터_10000건",
            user_name="Sujin",
            message_date=date(2026, 3, 19),
            chunk_text="[이슈] 장문 요청 처리 중 GPU 메모리 부족 발생\n[수정 및 결과] 배치 크기 제한을 적용했다",
            metadata={
                "entities": ["GPU", "메모리 부족", "OOM"],
                "chunk_kind": "analysis_flow",
                "issue_title": "장문 요청 처리 중 GPU 메모리 부족 발생",
                "assignee": "Sujin",
                "status": "완료",
                "flow_name": "수정 및 결과",
            },
        ),
        make_chunk(
            "issue-filter-overview",
            document_type="issue",
            vector_score=0.84,
            channel="이슈데이터_10000건",
            user_name="Jiho",
            message_date=date(2026, 3, 17),
            chunk_text="[이슈] 메타데이터 필터 적용 시 관련 문서 누락",
            metadata={
                "entities": ["메타데이터", "필터"],
                "chunk_kind": "overview",
                "issue_title": "메타데이터 필터 적용 시 관련 문서 누락",
                "assignee": "Jiho",
                "status": "보류",
            },
        ),
        make_chunk(
            "issue-embedding-overview",
            document_type="issue",
            vector_score=0.76,
            channel="이슈데이터_10000건",
            user_name="Minji",
            message_date=date(2026, 3, 20),
            chunk_text="[이슈] 임베딩 버전 불일치로 유사도 분포 이상",
            metadata={
                "entities": ["임베딩", "유사도"],
                "chunk_kind": "overview",
                "issue_title": "임베딩 버전 불일치로 유사도 분포 이상",
                "assignee": "Minji",
                "status": "완료",
            },
        ),
        make_chunk(
            "chat-gpu",
            document_type="chat",
            vector_score=0.62,
            channel="프로젝트C",
            user_name="박소율",
            message_date=date(2026, 3, 19),
            chunk_text="프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용",
            metadata={
                "entities": ["GPU", "메모리 부족"],
                "original_lines": ["프로젝트C 박소율: GPU 메모리 부족 대응 위해 배치 크기 제한 적용"],
            },
        ),
        make_chunk(
            "chat-generic",
            document_type="chat",
            vector_score=0.44,
            channel="개발팀",
            user_name="김민수",
            message_date=date(2026, 3, 18),
            chunk_text="개발팀 김민수: 회의록 공유 부탁드립니다",
            metadata={
                "entities": ["회의록"],
                "original_lines": ["개발팀 김민수: 회의록 공유 부탁드립니다"],
            },
        ),
        make_chunk(
            "chat-deploy",
            document_type="chat",
            vector_score=0.78,
            channel="프로젝트C",
            user_name="박소율",
            message_date=date(2024, 1, 5),
            chunk_text="프로젝트C 박소율: 서버 배포 380차 완료했습니다",
            metadata={
                "entities": ["서버", "배포"],
                "original_lines": ["프로젝트C 박소율: 서버 배포 380차 완료했습니다"],
            },
        ),
        make_chunk(
            "chat-backend",
            document_type="chat",
            vector_score=0.74,
            channel="백엔드개발",
            user_name="박동현",
            message_date=date(2026, 3, 20),
            chunk_text="백엔드개발 박동현: 운영 서버 반영은 오늘 진행합니다",
            metadata={
                "entities": ["운영 서버", "반영"],
                "original_lines": ["백엔드개발 박동현: 운영 서버 반영은 오늘 진행합니다"],
            },
        ),
    ]
