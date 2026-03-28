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
from app.services.workbook_parser import PayloadTooLargeError

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
            document_type=payload.document_type,
        )
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UploadFileResponse(document=document, ingest_summary=document.ingest_summary)


@router.post("/upload-file", response_model=UploadFileResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    default_access_scopes: str = Form("public"),
    source: str = Form("upload"),
    document_type: str = Form("auto"),
    container: ServiceContainer = Depends(get_container),
) -> UploadFileResponse:
    file_bytes = await file.read()

    try:
        document = await container.ingest.ingest_document(
            filename=file.filename or "uploaded.txt",
            file_bytes=file_bytes,
            default_access_scopes=parse_access_scopes(default_access_scopes),
            source=source,
            document_type=document_type,
            byte_limit=container.settings.api_issue_upload_max_bytes if document_type in ("auto", "issue") else None,
            row_limit=container.settings.api_issue_upload_max_rows if document_type in ("auto", "issue") else None,
        )
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return UploadFileResponse(document=document, ingest_summary=document.ingest_summary)


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
