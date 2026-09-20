from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from app.models.document import (
    DocumentRecord,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.document_service import (
    DocumentService,
)


router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


# =========================================================
# LIST DOCUMENTS
# =========================================================

@router.get(
    "",
    response_model=list[DocumentRecord],
)
async def list_documents(
) -> list[DocumentRecord]:
    """
    Return all persisted documents.
    """

    service = DocumentService()

    return service.list_documents()


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    """

    service = DocumentService()

    filename = file.filename

    if not filename:
        raise ValueError(
            "Uploaded file must have a filename."
        )

    safe_filename = (
        service.validate_filename(
            filename=filename,
        )
    )

    document_id = (
        service.create_document_id()
    )

    file_path = (
        service.save_uploaded_file(
            file_object=file.file,
            filename=safe_filename,
            document_id=document_id,
        )
    )

    document, chunks = (
        service.process_document(
            document_id=document_id,
            file_path=file_path,
            source=safe_filename,
        )
    )

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
        message=(
            "Document processed successfully. "
            f"Created {len(chunks)} chunk(s)."
        ),
    )


# =========================================================
# DOCUMENT STATUS
# =========================================================

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
)
async def get_document_status(
    document_id: str,
) -> DocumentStatusResponse:
    """
    Return persisted document status.
    """

    service = DocumentService()

    document = (
        service.get_document(
            document_id=document_id,
        )
    )

    chunks = (
        service.get_chunks(
            document_id=document_id,
        )
    )

    return DocumentStatusResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
        chunk_count=len(chunks),
        error_message=document.error_message,
    )


# =========================================================
# DELETE DOCUMENT
# =========================================================

@router.delete(
    "/{document_id}",
    response_model=dict,
)
async def delete_document(
    document_id: str,
) -> dict:
    """
    Delete a document and its persisted resources.

    Cleanup includes:

    - FAISS
    - Neo4j document evidence
    - chunks
    - original uploaded file
    - document metadata
    """

    service = DocumentService()

    result = (
        service.delete_document(
            document_id=document_id,
        )
    )

    return {
        "status": "deleted",
        **result,
    }