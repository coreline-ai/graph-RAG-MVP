from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import parse_access_scopes
from app.container import ServiceContainer, get_container
from app.schemas import (
    DocumentCreateRequest,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    UploadFileResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadFileResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreateRequest,
    container: ServiceContainer = Depends(get_container),
) -> UploadFileResponse:
    try:
        document = await container.ingest.ingest_document(
            filename=payload.filename,
            content=payload.content,
            default_access_scopes=payload.default_access_scopes,
            source=payload.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UploadFileResponse(document=document)


@router.post("/upload-file", response_model=UploadFileResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    default_access_scopes: str = Form("public"),
    source: str = Form("upload"),
    container: ServiceContainer = Depends(get_container),
) -> UploadFileResponse:
    try:
        content = (await file.read()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be UTF-8 encoded text.",
        ) from exc

    try:
        document = await container.ingest.ingest_document(
            filename=file.filename or "uploaded.txt",
            content=content,
            default_access_scopes=parse_access_scopes(default_access_scopes),
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UploadFileResponse(document=document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    container: ServiceContainer = Depends(get_container),
) -> DocumentListResponse:
    documents = await container.ingest.list_documents()
    return DocumentListResponse(documents=documents)


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> DocumentMetadata:
    document = await container.ingest.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> DocumentDeleteResponse:
    deleted = await container.ingest.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentDeleteResponse(document_id=document_id, deleted=True)
