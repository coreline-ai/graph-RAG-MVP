from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings, parse_access_scopes
from app.container import ServiceContainer


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a query against the GraphRAG pipeline.")
    parser.add_argument("question", help="Natural language question.")
    parser.add_argument(
        "--access-scopes",
        default="public",
        help="Comma-separated access scopes. Example: public,team-a",
    )
    parser.add_argument("--request-user", default=None, help="Optional requesting user id.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k retrieval limit.")
    args = parser.parse_args()

    settings = get_settings()
    container = ServiceContainer.create(settings)
    await container.startup()
    try:
        response = await container.retrieval.answer_query(
            question=args.question,
            access_scopes=parse_access_scopes(args.access_scopes),
            request_user=args.request_user,
            top_k=args.top_k,
        )
    finally:
        await container.shutdown()

    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
