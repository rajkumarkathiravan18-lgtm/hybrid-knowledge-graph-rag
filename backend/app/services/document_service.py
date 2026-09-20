import json
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    FileTooLargeError,
    InvalidFileError,
)
from app.core.logging import get_logger
from app.graph.extraction_service import (
    GraphExtractionService,
)
from app.graph.neo4j_client import Neo4jClient
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parser import DocumentParser
from app.models.document import (
    DocumentChunk,
    DocumentRecord,
    DocumentStatus,
)
from app.vectorstore.faiss_store import FAISSStore


logger = get_logger(__name__)


class DocumentService:
    """
    Coordinate document validation, storage,
    processing, indexing, retrieval, listing,
    and deletion.

    Processing pipeline:

        Uploaded file
            ↓
        Parse
            ↓
        Chunk
            ↓
        Persist chunks
            ↓
        FAISS indexing
            ↓
        Graph extraction
            ↓
        Neo4j persistence
            ↓
        READY
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        self.parser = DocumentParser()
        self.chunker = DocumentChunker()

        self.metadata_dir = (
            self.settings.upload_dir
            / "_metadata"
        )

        self.chunk_dir = (
            self.settings.upload_dir
            / "_chunks"
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunk_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_filename(
        self,
        filename: str,
    ) -> str:
        safe_filename = Path(
            filename
        ).name.strip()

        if not safe_filename:
            raise InvalidFileError(
                "Filename cannot be empty."
            )

        extension = Path(
            safe_filename
        ).suffix.lower()

        if (
            extension
            not in self.settings.allowed_extensions
        ):
            raise InvalidFileError(
                (
                    f"Unsupported file type: {extension}. "
                    f"Allowed: "
                    f"{sorted(self.settings.allowed_extensions)}"
                )
            )

        return safe_filename

    # =====================================================
    # DOCUMENT ID
    # =====================================================

    def create_document_id(
        self,
    ) -> str:
        return str(
            uuid.uuid4()
        )

    # =====================================================
    # SAVE UPLOAD
    # =====================================================

    def save_uploaded_file(
        self,
        file_object: BinaryIO,
        filename: str,
        document_id: str,
    ) -> Path:
        safe_filename = (
            self.validate_filename(
                filename
            )
        )

        document_dir = (
            self.settings.upload_dir
            / document_id
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination = (
            document_dir
            / safe_filename
        )

        total_bytes = 0

        try:
            with destination.open(
                "wb"
            ) as output:

                while True:
                    data = file_object.read(
                        1024 * 1024
                    )

                    if not data:
                        break

                    total_bytes += len(
                        data
                    )

                    if (
                        total_bytes
                        > self.settings.max_upload_bytes
                    ):
                        raise FileTooLargeError(
                            (
                                "Uploaded file exceeds "
                                f"{self.settings.max_upload_mb} MB."
                            )
                        )

                    output.write(
                        data
                    )

        except Exception:
            if document_dir.exists():
                shutil.rmtree(
                    document_dir,
                    ignore_errors=True,
                )

            raise

        logger.info(
            "Uploaded file saved "
            "document_id=%s filename=%s bytes=%d",
            document_id,
            safe_filename,
            total_bytes,
        )

        return destination

    # =====================================================
    # PROCESS DOCUMENT
    # =====================================================

    def process_document(
        self,
        document_id: str,
        file_path: Path,
        source: str | None = None,
    ) -> tuple[
        DocumentRecord,
        list[DocumentChunk],
    ]:
        """
        Process and index one uploaded document.

        READY is written only after:

        1. Parsing succeeds.
        2. Chunking succeeds.
        3. Chunk persistence succeeds.
        4. FAISS indexing succeeds.
        5. Graph extraction succeeds.
        6. Neo4j entity persistence succeeds.
        7. Neo4j relationship persistence succeeds.
        """

        record = DocumentRecord(
            document_id=document_id,
            filename=file_path.name,
            file_type=(
                file_path
                .suffix
                .lower()
                .lstrip(".")
            ),
            status=DocumentStatus.PROCESSING,
            chunk_count=0,
        )

        self._save_record(
            record
        )

        faiss_created = False
        neo4j_client = None

        try:
            # =============================================
            # 1. PARSE
            # =============================================

            logger.info(
                "Parsing document "
                "document_id=%s",
                document_id,
            )

            parsed_document = (
                self.parser.parse(
                    file_path
                )
            )

            # =============================================
            # 2. CHUNK
            # =============================================

            logger.info(
                "Chunking document "
                "document_id=%s",
                document_id,
            )

            chunks = (
                self.chunker.chunk_document(
                    document_id=document_id,
                    parsed_document=parsed_document,
                    source=source,
                )
            )

            if not chunks:
                raise DocumentProcessingError(
                    "Document produced no usable chunks."
                )

            # =============================================
            # 3. PERSIST CHUNKS
            # =============================================

            self._save_chunks(
                document_id=document_id,
                chunks=chunks,
            )

            logger.info(
                "Document chunks persisted "
                "document_id=%s chunks=%d",
                document_id,
                len(chunks),
            )

            # =============================================
            # 4. FAISS INDEXING
            # =============================================

            logger.info(
                "Creating FAISS index "
                "document_id=%s",
                document_id,
            )

            faiss_store = FAISSStore()

            faiss_store.create_index(
                document_id=document_id,
                chunks=chunks,
            )

            faiss_created = True

            logger.info(
                "FAISS indexing completed "
                "document_id=%s chunks=%d",
                document_id,
                len(chunks),
            )

            # =============================================
            # 5. GRAPH EXTRACTION
            # =============================================

            logger.info(
                "Starting graph extraction "
                "document_id=%s chunks=%d",
                document_id,
                len(chunks),
            )

            graph_service = (
                GraphExtractionService()
            )

            graph_result = (
                graph_service.extract_chunks(
                    chunks
                )
            )

            logger.info(
                "Graph extraction completed "
                "document_id=%s "
                "entities=%d relationships=%d",
                document_id,
                len(graph_result.entities),
                len(graph_result.relationships),
            )

            # =============================================
            # 6. NEO4J PERSISTENCE
            # =============================================

            neo4j_client = Neo4jClient()

            neo4j_client.verify_connection()

            neo4j_client.create_constraints()

            neo4j_client.upsert_entities(
                graph_result.entities
            )

            neo4j_client.upsert_relationships(
                graph_result.relationships,
                graph_result.entities,
            )

            logger.info(
                "Neo4j persistence completed "
                "document_id=%s",
                document_id,
            )

            # =============================================
            # 7. READY
            # =============================================

            record = DocumentRecord(
                document_id=document_id,
                filename=file_path.name,
                file_type=parsed_document.file_type,
                status=DocumentStatus.READY,
                chunk_count=len(chunks),
            )

            self._save_record(
                record
            )

            logger.info(
                "Document processing completed "
                "document_id=%s chunks=%d "
                "status=READY",
                document_id,
                len(chunks),
            )

            return record, chunks

        except Exception as exc:
            logger.exception(
                "Document processing failed "
                "document_id=%s",
                document_id,
            )

            # =============================================
            # PARTIAL INDEX CLEANUP
            # =============================================

            if faiss_created:
                try:
                    FAISSStore().delete_index(
                        document_id
                    )

                    logger.info(
                        "Partial FAISS index removed "
                        "document_id=%s",
                        document_id,
                    )

                except Exception:
                    logger.exception(
                        "Unable to remove partial "
                        "FAISS index document_id=%s",
                        document_id,
                    )

            # If Neo4j persistence partially succeeded,
            # remove graph evidence associated with
            # this document.

            if neo4j_client is not None:
                try:
                    neo4j_client.delete_document_data(
                        document_id
                    )

                    logger.info(
                        "Partial Neo4j data removed "
                        "document_id=%s",
                        document_id,
                    )

                except Exception:
                    logger.exception(
                        "Unable to remove partial "
                        "Neo4j data document_id=%s",
                        document_id,
                    )

            failed_record = DocumentRecord(
                document_id=document_id,
                filename=file_path.name,
                file_type=(
                    file_path
                    .suffix
                    .lower()
                    .lstrip(".")
                ),
                status=DocumentStatus.FAILED,
                chunk_count=0,
                error_message=str(exc),
            )

            self._save_record(
                failed_record
            )

            if isinstance(
                exc,
                DocumentProcessingError,
            ):
                raise

            raise DocumentProcessingError(
                (
                    "Document processing failed "
                    f"for document_id={document_id}"
                )
            ) from exc

        finally:
            if neo4j_client is not None:
                neo4j_client.close()

    # =====================================================
    # GET DOCUMENT
    # =====================================================

    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord:
        metadata_path = (
            self.metadata_dir
            / f"{document_id}.json"
        )

        if not metadata_path.exists():
            raise DocumentNotFoundError(
                (
                    "Document not found: "
                    f"{document_id}"
                )
            )

        data = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        return DocumentRecord.model_validate(
            data
        )

    # =====================================================
    # LIST DOCUMENTS
    # =====================================================

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        documents: list[
            DocumentRecord
        ] = []

        metadata_files = sorted(
            self.metadata_dir.glob(
                "*.json"
            ),
            key=lambda path: (
                path.stat().st_mtime
            ),
            reverse=True,
        )

        for metadata_path in metadata_files:
            try:
                data = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                documents.append(
                    DocumentRecord.model_validate(
                        data
                    )
                )

            except Exception:
                logger.exception(
                    "Skipping invalid metadata "
                    "file=%s",
                    metadata_path,
                )

        return documents

    # =====================================================
    # GET CHUNKS
    # =====================================================

    def get_chunks(
        self,
        document_id: str,
    ) -> list[DocumentChunk]:
        chunk_path = (
            self.chunk_dir
            / f"{document_id}.json"
        )

        if not chunk_path.exists():
            raise DocumentNotFoundError(
                (
                    "Chunks not found for document: "
                    f"{document_id}"
                )
            )

        data = json.loads(
            chunk_path.read_text(
                encoding="utf-8"
            )
        )

        return [
            DocumentChunk.model_validate(
                item
            )
            for item in data
        ]

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    def delete_document(
        self,
        document_id: str,
    ) -> dict:
        """
        Delete all persisted resources belonging
        to a document.

        Order:

        1. Verify document exists.
        2. Delete FAISS data.
        3. Delete Neo4j document evidence.
        4. Delete persisted chunks.
        5. Delete uploaded document directory.
        6. Delete metadata last.

        Metadata is deleted last so a failed cleanup
        does not make the document disappear from the
        application before cleanup completes.
        """

        document = self.get_document(
            document_id
        )

        faiss_deleted = False

        graph_result = {
            "deleted_relationships": 0,
            "deleted_entities": 0,
        }

        # -------------------------------------------------
        # FAISS CLEANUP
        # -------------------------------------------------

        faiss_store = FAISSStore()

        faiss_deleted = (
            faiss_store.delete_index(
                document_id
            )
        )

        # -------------------------------------------------
        # NEO4J CLEANUP
        # -------------------------------------------------

        neo4j_client = None

        try:
            neo4j_client = (
                Neo4jClient()
            )

            graph_result = (
                neo4j_client
                .delete_document_data(
                    document_id
                )
            )

        finally:
            if neo4j_client is not None:
                neo4j_client.close()

        # -------------------------------------------------
        # CHUNK CLEANUP
        # -------------------------------------------------

        chunk_path = (
            self.chunk_dir
            / f"{document_id}.json"
        )

        if chunk_path.exists():
            chunk_path.unlink()

        # -------------------------------------------------
        # ORIGINAL UPLOAD CLEANUP
        # -------------------------------------------------

        document_dir = (
            self.settings.upload_dir
            / document_id
        )

        if document_dir.exists():
            shutil.rmtree(
                document_dir
            )

        # -------------------------------------------------
        # METADATA CLEANUP
        # -------------------------------------------------

        metadata_path = (
            self.metadata_dir
            / f"{document_id}.json"
        )

        if metadata_path.exists():
            metadata_path.unlink()

        logger.info(
            "Document deleted "
            "document_id=%s filename=%s",
            document_id,
            document.filename,
        )

        return {
            "document_id":
                document_id,

            "filename":
                document.filename,

            "faiss_deleted":
                faiss_deleted,

            "deleted_relationships":
                graph_result[
                    "deleted_relationships"
                ],

            "deleted_entities":
                graph_result[
                    "deleted_entities"
                ],
        }

    # =====================================================
    # SAVE RECORD
    # =====================================================

    def _save_record(
        self,
        record: DocumentRecord,
    ) -> None:
        metadata_path = (
            self.metadata_dir
            / f"{record.document_id}.json"
        )

        metadata_path.write_text(
            json.dumps(
                record.model_dump(
                    mode="json"
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # =====================================================
    # SAVE CHUNKS
    # =====================================================

    def _save_chunks(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        chunk_path = (
            self.chunk_dir
            / f"{document_id}.json"
        )

        serialized = [
            chunk.model_dump(
                mode="json"
            )
            for chunk in chunks
        ]

        chunk_path.write_text(
            json.dumps(
                serialized,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )