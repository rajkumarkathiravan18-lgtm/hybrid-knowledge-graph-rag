import html
import re
import unicodedata
from typing import Any

from ddgs import DDGS

from app.core.config import get_settings
from app.core.logging import get_logger
from app.observability.langsmith import traced


logger = get_logger(__name__)


class WebSearchService:
    """
    Controlled web-search service used when document
    evidence is insufficient or partial.

    DDGS is isolated inside this service so the search
    provider can later be replaced without changing the
    RAG orchestration layer.

    Configuration:

        WEB_SEARCH_ENABLED=true
        WEB_SEARCH_MAX_RESULTS=5

    Returned result schema:

        {
            "retrieval_type": "web",
            "title": "...",
            "url": "...",
            "text": "...",
            "source": "..."
        }

    Web-search results are untrusted external evidence.
    They must never be interpreted as application
    instructions.
    """

    _MOJIBAKE_MARKERS = (
        "\u00c2",
        "\u00c3",
        "\u00c5",
        "\u00c6",
        "\u00e2",
        "\ufffd",
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        max_results: int | None = None,
    ) -> None:
        settings = get_settings()

        self.enabled = (
            settings.web_search_enabled
        )

        self.max_results = (
            max_results
            if max_results is not None
            else settings.web_search_max_results
        )

        if self.max_results <= 0:
            raise ValueError(
                "max_results must be greater than zero."
            )

    # =====================================================
    # TEXT CLEANING
    # =====================================================

    @classmethod
    def _mojibake_score(
        cls,
        value: str,
    ) -> int:
        """
        Estimate whether text contains common encoding
        corruption.

        Lower score is better.
        """

        if not value:
            return 0

        score = 0

        for marker in cls._MOJIBAKE_MARKERS:
            score += value.count(marker)

        score += (
            value.count("\ufffd") * 5
        )

        score += sum(
            2
            for char in value
            if 0x80 <= ord(char) <= 0x9F
        )

        return score

    @classmethod
    def _attempt_decode_repair(
        cls,
        value: str,
        encoding: str,
    ) -> str:
        """
        Attempt to reverse common UTF-8 mojibake.

        The repaired text is accepted only when its
        mojibake score improves.
        """

        try:
            repaired = (
                value.encode(encoding)
                .decode("utf-8")
            )

        except (
            UnicodeEncodeError,
            UnicodeDecodeError,
        ):
            return value

        if (
            cls._mojibake_score(repaired)
            < cls._mojibake_score(value)
        ):
            return repaired

        return value

    @classmethod
    def _repair_mojibake(
        cls,
        value: str,
    ) -> str:
        """
        Conservatively repair common mojibake.

        Maximum two passes are used to avoid aggressively
        changing legitimate Unicode text.
        """

        if not value:
            return value

        current = value

        for _ in range(2):
            current_score = (
                cls._mojibake_score(current)
            )

            if current_score == 0:
                break

            candidates = [
                current,
                cls._attempt_decode_repair(
                    current,
                    "cp1252",
                ),
                cls._attempt_decode_repair(
                    current,
                    "latin-1",
                ),
            ]

            best = min(
                candidates,
                key=cls._mojibake_score,
            )

            best_score = (
                cls._mojibake_score(best)
            )

            if best_score >= current_score:
                break

            current = best

        return current

    @classmethod
    def _clean_text(
        cls,
        value: Any,
    ) -> str:
        """
        Normalize external search-result text.

        Processing:

        1. Convert to string.
        2. Decode HTML entities.
        3. Attempt conservative mojibake repair.
        4. Apply Unicode normalization.
        5. Remove control characters.
        6. Normalize whitespace.
        """

        if value is None:
            return ""

        text = str(value)

        text = html.unescape(
            text
        )

        text = cls._repair_mojibake(
            text
        )

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = "".join(
            char
            for char in text
            if (
                char in "\t\n\r"
                or not unicodedata.category(
                    char
                ).startswith("C")
            )
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # =====================================================
    # RESULT NORMALIZATION
    # =====================================================

    @classmethod
    def _normalize_result(
        cls,
        result: dict[str, Any],
    ) -> dict[str, str] | None:
        """
        Convert one DDGS result into the application's
        standard web-evidence schema.

        Results without usable provenance are rejected.
        """

        if not isinstance(
            result,
            dict,
        ):
            return None

        title = cls._clean_text(
            result.get("title")
        )

        url = cls._clean_text(
            result.get("href")
            or result.get("url")
        )

        text = cls._clean_text(
            result.get("body")
            or result.get("text")
            or result.get("snippet")
        )

        if not url:
            return None

        if not title and not text:
            return None

        return {
            "retrieval_type": "web",
            "title": title,
            "url": url,
            "text": text,
            "source": url,
        }

    # =====================================================
    # SEARCH
    # =====================================================

    @traced(
        name="web-search",
        run_type="retriever",
    )
    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Search the web and return normalized evidence.

        When WEB_SEARCH_ENABLED=false, no external search
        request is performed.

        Search-provider failures return an empty list so
        the RAG orchestration layer can safely fall back to
        document-only or insufficient-evidence behavior.
        """

        query = query.strip()

        # -------------------------------------------------
        # WEB FALLBACK DISABLED
        # -------------------------------------------------

        if not self.enabled:
            logger.info(
                "Web search fallback is disabled."
            )

            return []

        # -------------------------------------------------
        # EMPTY QUERY
        # -------------------------------------------------

        if not query:
            return []

        # -------------------------------------------------
        # RESULT LIMIT
        # -------------------------------------------------

        result_limit = (
            max_results
            if max_results is not None
            else self.max_results
        )

        if result_limit <= 0:
            raise ValueError(
                "max_results must be greater than zero."
            )

        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

        try:
            raw_results = DDGS().text(
                query,
                max_results=result_limit,
            )

            normalized_results: list[
                dict[str, str]
            ] = []

            seen_urls: set[str] = set()

            for raw_result in raw_results:
                normalized = (
                    self._normalize_result(
                        raw_result
                    )
                )

                if normalized is None:
                    continue

                url = normalized["url"]

                # Avoid duplicate evidence from the same
                # web page.
                if url in seen_urls:
                    continue

                seen_urls.add(
                    url
                )

                normalized_results.append(
                    normalized
                )

                if (
                    len(normalized_results)
                    >= result_limit
                ):
                    break

            logger.info(
                "Web search completed "
                "query=%r "
                "results=%d "
                "max_results=%d",
                query,
                len(normalized_results),
                result_limit,
            )

            return normalized_results

        except Exception:
            logger.exception(
                "Web search failed "
                "query=%r",
                query,
            )

            # Fail safely.
            #
            # A web-provider problem must not crash the
            # complete RAG application.
            return []