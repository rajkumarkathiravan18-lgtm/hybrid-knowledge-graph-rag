from app.core.config import get_settings
from app.core.logging import get_logger
from app.observability.langsmith import traced


logger = get_logger(__name__)


class WebContextBuilder:
    """
    Convert normalized web-search results into bounded,
    clearly identified web evidence for the language model.

    Web content is external and untrusted.

    This builder:
    - preserves source URLs
    - preserves result titles
    - removes duplicate URLs
    - enforces the application's context-size limit
    - explicitly labels web evidence as untrusted data
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
    # FORMAT RESULT
    # =====================================================

    @staticmethod
    def _format_result(
        result: dict,
        position: int,
    ) -> str:
        """
        Format one web-search result.
        """

        title = (
            result.get("title")
            or "Untitled source"
        ).strip()

        url = (
            result.get("url")
            or result.get("source")
            or ""
        ).strip()

        text = (
            result.get("text")
            or ""
        ).strip()

        return (
            f"[WEB EVIDENCE {position}]\n"
            f"Evidence Type: WEB\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{text}"
        )

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    @traced(
        name="web-context-building",
        run_type="chain",
    )
    def build(
        self,
        results: list[dict],
    ) -> str:
        """
        Build bounded web context.

        Search results are treated strictly as external
        evidence, never as application instructions.
        """

        if not results:
            return ""

        context_parts: list[str] = []
        seen_urls: set[str] = set()

        current_length = 0
        evidence_number = 1

        for result in results:
            url = str(
                result.get("url")
                or result.get("source")
                or ""
            ).strip()

            text = str(
                result.get("text")
                or ""
            ).strip()

            if not url:
                continue

            if not text:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            formatted = self._format_result(
                result=result,
                position=evidence_number,
            )

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
            "Web context built results=%d "
            "characters=%d limit=%d",
            len(results),
            len(context),
            self.max_context_chars,
        )

        return context