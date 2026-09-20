from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    document_id: str

    filename: str

    file_type: str

    source: str | None = None


class DocumentChunk(BaseModel):
    document_id: str

    chunk_id: str

    text: str = Field(
        ...,
        min_length=1,
    )

    filename: str

    file_type: str

    page_number: int | None = None

    source: str | None = None

    chunk_index: int


class DocumentRecord(BaseModel):
    document_id: str

    filename: str

    file_type: str

    status: DocumentStatus

    chunk_count: int = 0

    error_message: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str

    filename: str

    status: DocumentStatus

    message: str


class DocumentStatusResponse(BaseModel):
    document_id: str

    filename: str

    status: DocumentStatus

    chunk_count: int = 0

    error_message: str | None = None