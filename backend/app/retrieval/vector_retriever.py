from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.observability.langsmith import traced
from app.vectorstore.faiss_store import FAISSStore


logger = get_logger(__name__)


class VectorRetriever:
    """
    High-level vector retrieval service.

    Responsibilities:
    - receive a document ID
    - receive a user query
    - call the persistent FAISS store
    - return normalized vector retrieval results

    FAISS implementation details remain inside FAISSStore.
    """

    def __init__(
        self,
        store: FAISSStore | None = None,
    ) -> None:
        settings = get_settings()

        self.store = store or FAISSStore()

        self.default_top_k = (
            settings.vector_top_k
        )

    @traced(
        name="vector-retrieval",
        run_type="retriever",
    )
    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve semantically relevant chunks
        from the document's persistent FAISS index.

        This operation is traced separately in LangSmith
        so vector retrieval can be inspected independently
        inside the Hybrid RAG trace.
        """

        document_id = document_id.strip()
        query = query.strip()

        if not document_id:
            raise RetrievalError(
                "Document ID cannot be empty."
            )

        if not query:
            return []

        effective_top_k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        if effective_top_k <= 0:
            return []

        try:
            results = self.store.search(
                document_id=document_id,
                query=query,
                top_k=effective_top_k,
            )

            normalized_results = []

            for result in results:
                normalized_results.append(
                    {
                        "retrieval_type":
                            "vector",

                        "document_id":
                            result.get(
                                "document_id"
                            ),

                        "chunk_id":
                            result.get(
                                "chunk_id"
                            ),

                        "text":
                            result.get(
                                "text",
                                "",
                            ),

                        "filename":
                            result.get(
                                "filename"
                            ),

                        "file_type":
                            result.get(
                                "file_type"
                            ),

                        "page_number":
                            result.get(
                                "page_number"
                            ),

                        "source":
                            result.get(
                                "source"
                            ),

                        "chunk_index":
                            result.get(
                                "chunk_index"
                            ),

                        "score":
                            float(
                                result.get(
                                    "score",
                                    0.0,
                                )
                            ),
                    }
                )

            logger.info(
                "Vector retrieval completed "
                "document_id=%s results=%d",
                document_id,
                len(normalized_results),
            )

            return normalized_results

        except RetrievalError:
            raise

        except Exception as exc:
            logger.exception(
                "Vector retrieval failed "
                "document_id=%s",
                document_id,
            )

            raise RetrievalError(
                "Vector retrieval failed."
            ) from exc