from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.metadata import router as metadata_router
from app.api.query import router as query_router
from app.config import Settings, get_settings
from app.container import ServiceContainer


def create_app(
    *,
    settings: Settings | None = None,
    container: ServiceContainer | object | None = None,
) -> FastAPI:
    resolved_settings = settings or getattr(container, "settings", None) or get_settings()
    static_dir = Path(__file__).resolve().parent / "static"
    provided_container = container

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.container = provided_container or ServiceContainer.create(resolved_settings)
        await app.state.container.startup()
        try:
            yield
        finally:
            await app.state.container.shutdown()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    index_path = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def root():
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(metadata_router)
    app.include_router(query_router)
    return app


app = create_app()
