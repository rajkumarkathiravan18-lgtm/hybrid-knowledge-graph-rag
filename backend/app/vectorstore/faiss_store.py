import json
import shutil
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.document import DocumentChunk
from app.vectorstore.embeddings import EmbeddingService


logger = get_logger(__name__)


class FAISSStore:
    """
    Persistent FAISS vector store.

    Each document gets its own FAISS index and metadata file.

    data/faiss/<document_id>/
        index.faiss
        metadata.json
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        settings = get_settings()

        self.embedding_service = (
            embedding_service
            or EmbeddingService()
        )

        self.base_dir = Path(
            settings.faiss_dir
        ).resolve()

        self.base_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # PATH HELPERS
    # =====================================================

    def _document_dir(
        self,
        document_id: str,
    ) -> Path:
        return self.base_dir / document_id

    def _index_path(
        self,
        document_id: str,
    ) -> Path:
        return (
            self._document_dir(document_id)
            / "index.faiss"
        )

    def _metadata_path(
        self,
        document_id: str,
    ) -> Path:
        return (
            self._document_dir(document_id)
            / "metadata.json"
        )

    # =====================================================
    # CREATE INDEX
    # =====================================================

    def create_index(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Embed document chunks and persist the FAISS index.
        """

        if not document_id.strip():
            raise VectorStoreError(
                "Document ID cannot be empty."
            )

        if not chunks:
            raise VectorStoreError(
                "Cannot create FAISS index "
                "without document chunks."
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        try:
            vectors = (
                self.embedding_service
                .embed_documents(texts)
            )

            if not vectors:
                raise VectorStoreError(
                    "Embedding service returned "
                    "no vectors."
                )

            if len(vectors) != len(chunks):
                raise VectorStoreError(
                    "Embedding count does not match "
                    "document chunk count."
                )

            matrix = np.asarray(
                vectors,
                dtype="float32",
            )

            if matrix.ndim != 2:
                raise VectorStoreError(
                    "Embedding matrix must "
                    "be two-dimensional."
                )

            faiss.normalize_L2(matrix)

            dimension = matrix.shape[1]

            index = faiss.IndexFlatIP(
                dimension
            )

            index.add(matrix)

            document_dir = (
                self._document_dir(
                    document_id
                )
            )

            document_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            faiss.write_index(
                index,
                str(
                    self._index_path(
                        document_id
                    )
                ),
            )

            metadata = []

            for position, chunk in enumerate(chunks):
                metadata.append(
                    {
                        "vector_position": position,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "filename": chunk.filename,
                        "file_type": chunk.file_type,
                        "page_number": chunk.page_number,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                    }
                )

            with self._metadata_path(
                document_id
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(
                "FAISS index created "
                "document_id=%s chunks=%d dimension=%d",
                document_id,
                len(chunks),
                dimension,
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            logger.exception(
                "Unable to create FAISS index "
                "document_id=%s",
                document_id,
            )

            raise VectorStoreError(
                "Unable to create FAISS index."
            ) from exc

    # =====================================================
    # LOAD INDEX
    # =====================================================

    def load_index(
        self,
        document_id: str,
    ):
        """
        Load an existing persisted FAISS index.
        """

        index_path = self._index_path(
            document_id
        )

        if not index_path.exists():
            raise VectorStoreError(
                f"FAISS index not found for "
                f"document: {document_id}"
            )

        try:
            return faiss.read_index(
                str(index_path)
            )

        except Exception as exc:
            logger.exception(
                "Unable to load FAISS index "
                "document_id=%s",
                document_id,
            )

            raise VectorStoreError(
                "Unable to load FAISS index."
            ) from exc

    # =====================================================
    # LOAD METADATA
    # =====================================================

    def load_metadata(
        self,
        document_id: str,
    ) -> list[dict]:
        """
        Load metadata corresponding to FAISS positions.
        """

        metadata_path = self._metadata_path(
            document_id
        )

        if not metadata_path.exists():
            raise VectorStoreError(
                f"FAISS metadata not found for "
                f"document: {document_id}"
            )

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise VectorStoreError(
                    "Invalid FAISS metadata format."
                )

            return data

        except VectorStoreError:
            raise

        except Exception as exc:
            logger.exception(
                "Unable to load FAISS metadata "
                "document_id=%s",
                document_id,
            )

            raise VectorStoreError(
                "Unable to load FAISS metadata."
            ) from exc

    # =====================================================
    # EXISTS
    # =====================================================

    def exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether both persistent FAISS files exist.
        """

        return (
            self._index_path(
                document_id
            ).exists()
            and
            self._metadata_path(
                document_id
            ).exists()
        )

    # =====================================================
    # DELETE INDEX
    # =====================================================

    def delete_index(
        self,
        document_id: str,
    ) -> bool:
        """
        Delete all persisted FAISS data for a document.

        Returns True when a document FAISS directory
        existed and was removed.

        Returns False when no FAISS directory existed.
        """

        document_id = document_id.strip()

        if not document_id:
            raise VectorStoreError(
                "Document ID cannot be empty."
            )

        document_dir = (
            self._document_dir(
                document_id
            )
        )

        if not document_dir.exists():
            return False

        try:
            shutil.rmtree(
                document_dir
            )

            logger.info(
                "FAISS data deleted "
                "document_id=%s",
                document_id,
            )

            return True

        except Exception as exc:
            logger.exception(
                "Unable to delete FAISS data "
                "document_id=%s",
                document_id,
            )

            raise VectorStoreError(
                "Unable to delete FAISS data."
            ) from exc

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search a persisted FAISS index.

        Only the query is embedded here.
        Document chunks are not re-embedded.
        """

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            return []

        if not self.exists(document_id):
            raise VectorStoreError(
                f"FAISS index does not exist for "
                f"document: {document_id}"
            )

        try:
            index = self.load_index(
                document_id
            )

            metadata = self.load_metadata(
                document_id
            )

            if index.ntotal == 0:
                return []

            query_vector = (
                self.embedding_service
                .embed_query(query)
            )

            query_matrix = np.asarray(
                [query_vector],
                dtype="float32",
            )

            if query_matrix.ndim != 2:
                raise VectorStoreError(
                    "Invalid query embedding shape."
                )

            if query_matrix.shape[1] != index.d:
                raise VectorStoreError(
                    "Query embedding dimension does "
                    "not match FAISS index dimension."
                )

            faiss.normalize_L2(
                query_matrix
            )

            search_k = min(
                top_k,
                index.ntotal,
            )

            scores, positions = index.search(
                query_matrix,
                search_k,
            )

            results = []

            for score, position in zip(
                scores[0],
                positions[0],
            ):
                position = int(position)

                if position < 0:
                    continue

                if position >= len(metadata):
                    logger.warning(
                        "FAISS returned metadata position "
                        "outside metadata range "
                        "position=%d",
                        position,
                    )
                    continue

                item = metadata[
                    position
                ].copy()

                item["score"] = float(
                    score
                )

                item[
                    "retrieval_type"
                ] = "vector"

                results.append(
                    item
                )

            logger.info(
                "FAISS search completed "
                "document_id=%s results=%d",
                document_id,
                len(results),
            )

            return results

        except VectorStoreError:
            raise

        except Exception as exc:
            logger.exception(
                "FAISS search failed "
                "document_id=%s",
                document_id,
            )

            raise VectorStoreError(
                "Unable to search FAISS index."
            ) from exc