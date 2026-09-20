from app.core.config import get_settings
from app.core.logging import get_logger
from app.observability.langsmith import traced


logger = get_logger(__name__)


class ContextBuilder:
    """
    Converts ranked hybrid retrieval results into
    controlled context for the language model.

    Responsibilities:
    - preserve retrieval ranking
    - preserve document/chunk provenance
    - distinguish vector evidence from graph evidence
    - remove duplicate evidence
    - enforce maximum context size
    """

    def __init__(
        self,
        max_context_chars: int | None = None,
    ) -> None:
        settings = get_settings()

        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else settings.max_context_chars
        )

        if self.max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be greater than 0."
            )

    # =====================================================
    # VECTOR FORMAT
    # =====================================================

    @staticmethod
    def _format_vector_result(
        result: dict,
        position: int,
    ) -> str:
        """
        Format one vector retrieval result.
        """

        document_id = (
            result.get("document_id")
            or "unknown"
        )

        chunk_id = (
            result.get("chunk_id")
            or "unknown"
        )

        filename = (
            result.get("filename")
            or "unknown"
        )

        page_number = result.get(
            "page_number"
        )

        score = float(
            result.get(
                "hybrid_score",
                0.0,
            )
        )

        text = (
            result.get("text")
            or ""
        ).strip()

        page_display = (
            str(page_number)
            if page_number is not None
            else "N/A"
        )

        return (
            f"[EVIDENCE {position}]\n"
            f"Retrieval Type: VECTOR\n"
            f"Document ID: {document_id}\n"
            f"Filename: {filename}\n"
            f"Page: {page_display}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Hybrid Score: {score:.4f}\n"
            f"Content:\n{text}"
        )

    # =====================================================
    # GRAPH FORMAT
    # =====================================================

    @staticmethod
    def _format_graph_result(
        result: dict,
        position: int,
    ) -> str:
        """
        Format one knowledge-graph result.
        """

        document_id = (
            result.get("document_id")
            or "unknown"
        )

        chunk_id = (
            result.get("chunk_id")
            or "unknown"
        )

        source = (
            result.get("graph_source")
            or "unknown"
        )

        relationship = (
            result.get(
                "graph_relationship"
            )
            or "RELATED_TO"
        )

        target = (
            result.get("graph_target")
            or "unknown"
        )

        evidence = (
            result.get(
                "graph_evidence"
            )
            or result.get("text")
            or ""
        ).strip()

        score = float(
            result.get(
                "hybrid_score",
                0.0,
            )
        )

        return (
            f"[EVIDENCE {position}]\n"
            f"Retrieval Type: GRAPH\n"
            f"Document ID: {document_id}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Hybrid Score: {score:.4f}\n"
            f"Relationship: "
            f"{source} --{relationship}--> {target}\n"
            f"Supporting Evidence:\n{evidence}"
        )

    # =====================================================
    # DEDUPLICATION KEY
    # =====================================================

    @staticmethod
    def _evidence_key(
        result: dict,
    ) -> tuple:
        """
        Generate a stable deduplication key.
        """

        retrieval_type = result.get(
            "retrieval_type"
        )

        if retrieval_type == "graph":
            return (
                "graph",
                result.get("document_id"),
                result.get("chunk_id"),
                result.get("graph_source"),
                result.get(
                    "graph_relationship"
                ),
                result.get("graph_target"),
            )

        return (
            "vector",
            result.get("document_id"),
            result.get("chunk_id"),
        )

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    @traced(
        name="context-building",
        run_type="chain",
    )
    def build(
        self,
        results: list[dict],
    ) -> str:
        """
        Build the final bounded context string.

        Results are expected to already be ranked by
        HybridRetriever.

        This operation is traced independently in LangSmith
        so the final context passed toward generation can be
        inspected separately from retrieval.
        """

        if not results:
            return (
                "No relevant evidence was retrieved "
                "from the uploaded documents."
            )

        context_parts = []

        seen = set()

        current_length = 0
        evidence_number = 1

        for result in results:

            key = self._evidence_key(
                result
            )

            if key in seen:
                continue

            seen.add(key)

            retrieval_type = result.get(
                "retrieval_type"
            )

            if retrieval_type == "graph":
                formatted = (
                    self._format_graph_result(
                        result,
                        evidence_number,
                    )
                )

            else:
                formatted = (
                    self._format_vector_result(
                        result,
                        evidence_number,
                    )
                )

            if not formatted.strip():
                continue

            separator = (
                "\n\n"
                if context_parts
                else ""
            )

            required_chars = (
                len(separator)
                + len(formatted)
            )

            remaining_chars = (
                self.max_context_chars
                - current_length
            )

            if remaining_chars <= 0:
                break

            # If the complete evidence block fits,
            # include it normally.
            if required_chars <= remaining_chars:

                if separator:
                    context_parts.append(
                        separator
                    )

                context_parts.append(
                    formatted
                )

                current_length += (
                    required_chars
                )

                evidence_number += 1

                continue

            # Do not append a tiny unusable fragment.
            if remaining_chars < 200:
                break

            if separator:
                context_parts.append(
                    separator
                )

                remaining_chars -= len(
                    separator
                )

            truncation_marker = (
                "\n[TRUNCATED DUE TO CONTEXT LIMIT]"
            )

            available = (
                remaining_chars
                - len(truncation_marker)
            )

            if available > 0:
                context_parts.append(
                    formatted[:available]
                    + truncation_marker
                )

            break

        context = "".join(
            context_parts
        ).strip()

        logger.info(
            "Context built results=%d "
            "characters=%d limit=%d",
            len(results),
            len(context),
            self.max_context_chars,
        )

        return context