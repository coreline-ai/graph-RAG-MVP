#!/usr/bin/env python3
"""Offline batch script: detect entity communities and store in Neo4j.

Usage:
    python scripts/detect_communities.py [--with-summaries]

Requires:
    - Neo4j with GDS plugin enabled
    - COMMUNITY_DETECTION_ENABLED=true in .env (or passed as flag)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, ".")

from app.adapters.codex_proxy import CodexProxyClient
from app.adapters.neo4j_store import Neo4jStore
from app.config import get_settings
from app.services.community_detector import CommunityDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)


async def main(with_summaries: bool = False) -> None:
    settings = get_settings()
    neo4j = Neo4jStore(settings)
    codex_proxy = CodexProxyClient(settings) if with_summaries else None

    detector = CommunityDetector(
        settings=settings,
        neo4j=neo4j,
        codex_proxy=codex_proxy,
    )

    try:
        count = await detector.detect_and_store()
        _log.info("Done. Stored %d communities.", count)
    finally:
        await neo4j.close()
        if codex_proxy:
            await codex_proxy.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect entity communities")
    parser.add_argument("--with-summaries", action="store_true", help="Generate LLM summaries for communities")
    args = parser.parse_args()
    asyncio.run(main(with_summaries=args.with_summaries))
