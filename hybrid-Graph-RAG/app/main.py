import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.insights import router as insights_router
from app.api.messages import router as messages_router
from app.api.search import router as search_router
from app.api.ui import router as ui_router
from app.repositories.insights_repo import InsightsRepository
from app.repositories.neo4j_client import Neo4jClient
from app.repositories.search_repo import SearchRepository
from app.services.embedder import BgeM3Embedder
from app.services.insights_service import InsightsService
from app.services.search_service import SearchService
from app.settings import Settings, get_settings


logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(settings.log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def _attach_runtime(app: FastAPI, settings: Settings) -> None:
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    neo4j_client.verify_connectivity()

    search_repo = SearchRepository(neo4j_client)
    insights_repo = InsightsRepository(neo4j_client)
    embedder = BgeM3Embedder(settings=settings)

    app.state.neo4j_client = neo4j_client
    app.state.search_service = SearchService(search_repo=search_repo, embedder=embedder)
    app.state.insights_service = InsightsService(insights_repo=insights_repo)


def _close_runtime(app: FastAPI) -> None:
    neo4j_client = getattr(app.state, "neo4j_client", None)
    if neo4j_client is not None:
        neo4j_client.close()


def create_app(settings: Settings | None = None, enable_runtime: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if enable_runtime:
            _attach_runtime(app, settings)
        yield
        if enable_runtime:
            _close_runtime(app)

    app = FastAPI(title="Hybrid GraphRAG", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_: Request, exc: RuntimeError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(messages_router)
    app.include_router(insights_router)
    app.include_router(ui_router)
    return app


app = create_app()
