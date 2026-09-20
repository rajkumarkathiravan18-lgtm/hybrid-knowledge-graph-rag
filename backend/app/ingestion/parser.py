from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import (
    DocumentProcessingError,
    InvalidFileError,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """
    Represents one parsed unit from a document.

    For PDFs, one ParsedPage represents one PDF page.

    For TXT and Markdown files, the complete document is
    represented as one ParsedPage with page_number=None.
    """

    text: str
    page_number: int | None


@dataclass
class ParsedDocument:
    """
    Parsed document before cleaning and chunking.
    """

    filename: str
    file_type: str
    pages: list[ParsedPage]


class DocumentParser:
    """
    Parse supported documents into plain text while preserving
    useful page metadata.

    Supported:
        PDF
        TXT
        Markdown
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
    }

    def parse(
        self,
        file_path: str | Path,
    ) -> ParsedDocument:
        path = Path(file_path)

        if not path.exists():
            raise InvalidFileError(
                f"File does not exist: {path.name}"
            )

        if not path.is_file():
            raise InvalidFileError(
                f"Path is not a file: {path.name}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise InvalidFileError(
                (
                    f"Unsupported file type: {extension}. "
                    "Supported types: .pdf, .txt, .md"
                )
            )

        logger.info(
            "Parsing document filename=%s type=%s",
            path.name,
            extension,
        )

        try:
            if extension == ".pdf":
                pages = self._parse_pdf(path)

            elif extension in {".txt", ".md"}:
                pages = self._parse_text_file(path)

            else:
                raise InvalidFileError(
                    f"Unsupported file type: {extension}"
                )

        except InvalidFileError:
            raise

        except Exception as exc:
            logger.exception(
                "Document parsing failed filename=%s",
                path.name,
            )

            raise DocumentProcessingError(
                f"Unable to parse document: {path.name}"
            ) from exc

        if not pages:
            raise DocumentProcessingError(
                f"No readable content found in: {path.name}"
            )

        if not any(page.text.strip() for page in pages):
            raise DocumentProcessingError(
                (
                    f"No extractable text found in: {path.name}. "
                    "The document may be empty or image-only."
                )
            )

        logger.info(
            "Document parsed filename=%s parsed_units=%d",
            path.name,
            len(pages),
        )

        return ParsedDocument(
            filename=path.name,
            file_type=extension.lstrip("."),
            pages=pages,
        )

    def _parse_pdf(
        self,
        path: Path,
    ) -> list[ParsedPage]:
        """
        Extract text page-by-page from a PDF.

        Page numbering exposed to users starts from 1.
        """

        reader = PdfReader(str(path))

        pages: list[ParsedPage] = []

        for index, pdf_page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = pdf_page.extract_text() or ""

            except Exception as exc:
                logger.warning(
                    "Unable to extract PDF page filename=%s page=%d error=%s",
                    path.name,
                    index,
                    type(exc).__name__,
                )

                text = ""

            pages.append(
                ParsedPage(
                    text=text,
                    page_number=index,
                )
            )

        return pages

    def _parse_text_file(
        self,
        path: Path,
    ) -> list[ParsedPage]:
        """
        Read UTF-8 TXT or Markdown files.
        """

        try:
            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed filename=%s; "
                "retrying with UTF-8 BOM support",
                path.name,
            )

            try:
                text = path.read_text(
                    encoding="utf-8-sig"
                )

            except UnicodeDecodeError as exc:
                raise DocumentProcessingError(
                    (
                        f"Unable to decode {path.name}. "
                        "Please provide a UTF-8 encoded file."
                    )
                ) from exc

        return [
            ParsedPage(
                text=text,
                page_number=None,
            )
        ]