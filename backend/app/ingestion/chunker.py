from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.cleaner import clean_text
from app.ingestion.parser import ParsedDocument
from app.models.document import DocumentChunk


logger = get_logger(__name__)


class DocumentChunker:
    """
    Convert parsed document pages into retrieval-ready chunks
    while preserving document and page metadata.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunk_size
        )

        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

    def chunk_document(
        self,
        document_id: str,
        parsed_document: ParsedDocument,
        source: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Clean and chunk an entire parsed document.

        Returns:
            list[DocumentChunk]
        """

        chunks: list[DocumentChunk] = []

        global_chunk_index = 0

        for page in parsed_document.pages:
            cleaned_text = clean_text(
                page.text
            )

            if not cleaned_text:
                continue

            page_chunks = self._split_text(
                cleaned_text
            )

            for chunk_text in page_chunks:
                chunk_id = (
                    f"{document_id}-chunk-"
                    f"{global_chunk_index:05d}"
                )

                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    filename=parsed_document.filename,
                    file_type=parsed_document.file_type,
                    page_number=page.page_number,
                    source=source,
                    chunk_index=global_chunk_index,
                )

                chunks.append(chunk)

                global_chunk_index += 1

        logger.info(
            "Document chunking completed "
            "document_id=%s filename=%s chunks=%d",
            document_id,
            parsed_document.filename,
            len(chunks),
        )

        return chunks

    def _split_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into overlapping chunks.

        The splitter attempts to finish chunks at sensible
        boundaries where possible.
        """

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(
                start + self.chunk_size,
                text_length,
            )

            end = target_end

            if target_end < text_length:
                end = self._find_split_position(
                    text=text,
                    start=start,
                    target_end=target_end,
                )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            # Guarantee forward movement.
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _find_split_position(
        self,
        text: str,
        start: int,
        target_end: int,
    ) -> int:
        """
        Search backwards from the target end for a useful
        semantic boundary.

        Priority:
            paragraph
            newline
            sentence
            whitespace
            hard character boundary
        """

        search_start = max(
            start,
            target_end - 250,
        )

        search_area = text[
            search_start:target_end
        ]

        boundaries = [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
        ]

        for boundary in boundaries:
            position = search_area.rfind(
                boundary
            )

            if position != -1:
                absolute_position = (
                    search_start
                    + position
                    + len(boundary)
                )

                if absolute_position > start:
                    return absolute_position

        return target_end