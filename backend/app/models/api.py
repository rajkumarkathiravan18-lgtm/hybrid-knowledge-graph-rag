from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# HEALTH
# ============================================================


class HealthResponse(BaseModel):
    status: str
    application: str
    environment: str


# ============================================================
# RAG SOURCES
# ============================================================


class SourceReference(BaseModel):
    """
    Source reference for vector/document evidence.
    """

    document_id: str

    filename: str | None = None

    chunk_id: str | None = None

    page_number: int | None = None

    score: float | None = None


class GraphEvidence(BaseModel):
    """
    Structured evidence returned from the Neo4j
    knowledge graph.
    """

    source: str

    relationship: str

    target: str

    evidence: str | None = None

    document_id: str | None = None

    chunk_id: str | None = None

    score: float | None = None


class WebSource(BaseModel):
    """
    External web source used by controlled web fallback.

    Web sources are kept separate from uploaded-document
    sources so provenance remains explicit.
    """

    title: str | None = None

    url: str

    snippet: str | None = None


# ============================================================
# CHAT
# ============================================================


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    document_id: str = Field(
        ...,
        min_length=1,
    )

    # Optional for backward compatibility.
    #
    # If omitted, the backend creates a new persistent
    # conversation.
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
    )


class ChatResponse(BaseModel):
    answer: str

    document_id: str

    # Returned so the frontend can continue the same
    # persistent conversation.
    conversation_id: str | None = None

    # Indicates where the answer was grounded.
    #
    # Supported values:
    #
    # document
    # web
    # document+web
    # insufficient
    source_type: str = "document"

    # Vector/document sources.
    sources: list[SourceReference] = Field(
        default_factory=list
    )

    # Neo4j relationship evidence.
    graph_context: list[GraphEvidence] = Field(
        default_factory=list
    )

    # External sources used by controlled web fallback.
    web_sources: list[WebSource] = Field(
        default_factory=list
    )


# ============================================================
# PERSISTENT CONVERSATION MEMORY
# ============================================================


class ConversationMessage(BaseModel):
    message_id: str

    conversation_id: str

    role: str

    content: str

    source_type: str | None = None

    created_at: str


class ConversationSummary(BaseModel):
    conversation_id: str

    document_id: str | None = None

    title: str

    created_at: str

    updated_at: str


class ConversationDetail(BaseModel):
    conversation_id: str

    document_id: str | None = None

    title: str

    created_at: str

    updated_at: str

    messages: list[ConversationMessage] = Field(
        default_factory=list
    )


# ============================================================
# ERROR
# ============================================================


class ErrorResponse(BaseModel):
    error: str

    message: str

    details: dict[str, Any] = Field(
        default_factory=dict
    )