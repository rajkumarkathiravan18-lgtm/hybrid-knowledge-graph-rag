from fastapi import APIRouter

from app.models.api import (
    ChatRequest,
    ChatResponse,
    GraphEvidence,
    SourceReference,
    WebSource,
)
from app.services.chat_service import ChatService


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Chat"],
)


# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Run a question through the complete Hybrid Knowledge
    Graph RAG pipeline with persistent conversation memory,
    controlled web fallback, and explicit source provenance.

    Public response provenance rules:

        source_type = document
            -> vector/document sources
            -> graph evidence
            -> no web sources

        source_type = web
            -> no document sources
            -> no graph evidence
            -> web sources

        source_type = document+web
            -> document sources
            -> graph evidence
            -> web sources

        source_type = insufficient
            -> no answer sources

    Internal retrieval candidates that were rejected by the
    evidence-sufficiency stage are intentionally NOT exposed
    as answer sources.
    """

    service = ChatService()

    try:
        # ====================================================
        # CHAT SERVICE
        # ====================================================

        result = await service.chat(
            document_id=request.document_id,
            question=request.question,
            conversation_id=request.conversation_id,
        )

        source_type = str(
            result.get(
                "source_type",
                "insufficient",
            )
        ).strip()

        retrieval_results = result.get(
            "retrieval_results",
            [],
        )

        web_results = result.get(
            "web_results",
            [],
        )

        # ====================================================
        # RESPONSE COLLECTIONS
        # ====================================================

        sources: list[SourceReference] = []

        graph_context: list[GraphEvidence] = []

        web_sources: list[WebSource] = []

        # ====================================================
        # DOCUMENT / GRAPH PROVENANCE
        # ====================================================

        include_document_sources = (
            source_type
            in {
                "document",
                "document+web",
            }
        )

        if include_document_sources:

            seen_vector_sources: set[
                tuple[str, str | None]
            ] = set()

            seen_graph_evidence: set[
                tuple[
                    str,
                    str,
                    str,
                    str | None,
                ]
            ] = set()

            for item in retrieval_results:

                retrieval_type = item.get(
                    "retrieval_type"
                )

                # --------------------------------------------
                # VECTOR RESULT
                # --------------------------------------------

                if retrieval_type == "vector":

                    document_id = str(
                        item.get(
                            "document_id"
                        )
                        or request.document_id
                    )

                    chunk_id = item.get(
                        "chunk_id"
                    )

                    vector_key = (
                        document_id,
                        chunk_id,
                    )

                    if (
                        vector_key
                        in seen_vector_sources
                    ):
                        continue

                    seen_vector_sources.add(
                        vector_key
                    )

                    sources.append(
                        SourceReference(
                            document_id=document_id,
                            filename=item.get(
                                "filename"
                            ),
                            chunk_id=chunk_id,
                            page_number=item.get(
                                "page_number"
                            ),
                            score=item.get(
                                "hybrid_score",
                                item.get(
                                    "score"
                                ),
                            ),
                        )
                    )

                # --------------------------------------------
                # GRAPH RESULT
                # --------------------------------------------

                elif retrieval_type == "graph":

                    graph_source = item.get(
                        "graph_source"
                    )

                    graph_relationship = item.get(
                        "graph_relationship"
                    )

                    graph_target = item.get(
                        "graph_target"
                    )

                    graph_evidence = item.get(
                        "graph_evidence"
                    )

                    if not (
                        graph_source
                        and graph_relationship
                        and graph_target
                    ):
                        continue

                    graph_key = (
                        str(graph_source),
                        str(graph_relationship),
                        str(graph_target),
                        (
                            str(graph_evidence)
                            if graph_evidence
                            else None
                        ),
                    )

                    if (
                        graph_key
                        in seen_graph_evidence
                    ):
                        continue

                    seen_graph_evidence.add(
                        graph_key
                    )

                    graph_context.append(
                        GraphEvidence(
                            source=str(
                                graph_source
                            ),
                            relationship=str(
                                graph_relationship
                            ),
                            target=str(
                                graph_target
                            ),
                            evidence=(
                                str(graph_evidence)
                                if graph_evidence
                                else None
                            ),
                            document_id=item.get(
                                "document_id"
                            ),
                            chunk_id=item.get(
                                "chunk_id"
                            ),
                            score=item.get(
                                "hybrid_score"
                            ),
                        )
                    )

        # ====================================================
        # WEB PROVENANCE
        # ====================================================

        include_web_sources = (
            source_type
            in {
                "web",
                "document+web",
            }
        )

        if include_web_sources:

            seen_web_urls: set[str] = set()

            for item in web_results:

                url = str(
                    item.get("url")
                    or item.get("source")
                    or ""
                ).strip()

                if not url:
                    continue

                if url in seen_web_urls:
                    continue

                seen_web_urls.add(
                    url
                )

                title = str(
                    item.get("title")
                    or ""
                ).strip()

                snippet = str(
                    item.get("text")
                    or ""
                ).strip()

                web_sources.append(
                    WebSource(
                        title=(
                            title
                            if title
                            else None
                        ),
                        url=url,
                        snippet=(
                            snippet
                            if snippet
                            else None
                        ),
                    )
                )

        # ====================================================
        # RESPONSE
        # ====================================================

        return ChatResponse(
            answer=result.get(
                "answer",
                "",
            ),
            document_id=request.document_id,
            conversation_id=result.get(
                "conversation_id"
            ),
            source_type=source_type,
            sources=sources,
            graph_context=graph_context,
            web_sources=web_sources,
        )

    finally:
        service.close()