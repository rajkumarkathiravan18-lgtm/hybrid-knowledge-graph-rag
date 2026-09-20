from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.graph_retriever import GraphRetriever
from app.observability.langsmith import traced
from app.retrieval.vector_retriever import VectorRetriever


logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines vector retrieval and knowledge-graph retrieval.

    Vector retrieval:
        semantic similarity from FAISS

    Graph retrieval:
        structured relationship evidence from Neo4j

    Final output:
        ranked, deduplicated hybrid evidence

    LangSmith:
        the complete retrieval operation is traced as
        "hybrid-retrieval".
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        settings = get_settings()

        self.vector_retriever = (
            vector_retriever
            or VectorRetriever()
        )

        self.graph_retriever = (
            graph_retriever
            or GraphRetriever()
        )

        self.vector_weight = (
            settings.vector_weight
        )

        self.graph_weight = (
            settings.graph_weight
        )

        self.vector_top_k = (
            settings.vector_top_k
        )

        self.graph_top_k = (
            settings.graph_top_k
        )

    # =====================================================
    # VECTOR NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize_vector_scores(
        results: list[dict],
    ) -> list[dict]:
        """
        Normalize vector scores into the range 0..1.

        FAISS IndexFlatIP returns inner-product similarity.
        Because our vectors are L2-normalized, this behaves
        like cosine similarity.

        Min/max normalization is applied only across the
        current result set.
        """

        if not results:
            return []

        scores = [
            float(
                result.get(
                    "score",
                    0.0,
                )
            )
            for result in results
        ]

        minimum = min(scores)
        maximum = max(scores)

        normalized_results = []

        for result in results:
            item = result.copy()

            raw_score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            if maximum == minimum:
                normalized_score = 1.0
            else:
                normalized_score = (
                    (raw_score - minimum)
                    / (maximum - minimum)
                )

            item["raw_score"] = raw_score
            item["normalized_score"] = (
                normalized_score
            )

            normalized_results.append(
                item
            )

        return normalized_results

    # =====================================================
    # VECTOR RESULT CONVERSION
    # =====================================================

    def _prepare_vector_results(
        self,
        results: list[dict],
    ) -> list[dict]:
        """
        Convert vector results into the common hybrid
        retrieval result structure.
        """

        normalized = (
            self._normalize_vector_scores(
                results
            )
        )

        prepared = []

        for result in normalized:
            hybrid_score = (
                result["normalized_score"]
                * self.vector_weight
            )

            prepared.append(
                {
                    "retrieval_type":
                        "vector",

                    "hybrid_score":
                        hybrid_score,

                    "raw_score":
                        result["raw_score"],

                    "normalized_score":
                        result["normalized_score"],

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

                    "graph_source":
                        None,

                    "graph_relationship":
                        None,

                    "graph_target":
                        None,

                    "graph_evidence":
                        None,
                }
            )

        return prepared

    # =====================================================
    # GRAPH RESULT CONVERSION
    # =====================================================

    def _prepare_graph_results(
        self,
        results: list[dict],
    ) -> list[dict]:
        """
        Convert graph evidence into the same common
        structure used by vector results.

        Graph evidence receives the configured graph
        weight because it has already passed structured
        graph matching.
        """

        prepared = []

        for result in results:
            prepared.append(
                {
                    "retrieval_type":
                        "graph",

                    "hybrid_score":
                        self.graph_weight,

                    "raw_score":
                        None,

                    "normalized_score":
                        1.0,

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
                            "evidence",
                            "",
                        ),

                    "filename":
                        None,

                    "file_type":
                        None,

                    "page_number":
                        None,

                    "source":
                        None,

                    "chunk_index":
                        None,

                    "matched_entity":
                        result.get(
                            "matched_entity"
                        ),

                    "graph_source":
                        result.get(
                            "source"
                        ),

                    "graph_source_type":
                        result.get(
                            "source_type"
                        ),

                    "graph_relationship":
                        result.get(
                            "relationship"
                        ),

                    "graph_target":
                        result.get(
                            "target"
                        ),

                    "graph_target_type":
                        result.get(
                            "target_type"
                        ),

                    "graph_evidence":
                        result.get(
                            "evidence"
                        ),
                }
            )

        return prepared

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    @staticmethod
    def _deduplicate(
        results: list[dict],
    ) -> list[dict]:
        """
        Remove duplicate retrieval evidence.

        Vector evidence is primarily deduplicated by
        document + chunk.

        Graph evidence additionally includes the graph
        relationship so multiple distinct graph facts from
        the same chunk are preserved.
        """

        deduplicated = []
        seen = set()

        for result in results:

            if (
                result.get(
                    "retrieval_type"
                )
                == "graph"
            ):
                key = (
                    "graph",
                    result.get(
                        "document_id"
                    ),
                    result.get(
                        "chunk_id"
                    ),
                    result.get(
                        "graph_source"
                    ),
                    result.get(
                        "graph_relationship"
                    ),
                    result.get(
                        "graph_target"
                    ),
                )

            else:
                key = (
                    "vector",
                    result.get(
                        "document_id"
                    ),
                    result.get(
                        "chunk_id"
                    ),
                )

            if key in seen:
                continue

            seen.add(key)

            deduplicated.append(
                result
            )

        return deduplicated

    # =====================================================
    # RETRIEVE
    # =====================================================

    @traced(
        name="hybrid-retrieval",
        run_type="retriever",
    )
    def retrieve(
        self,
        document_id: str,
        query: str,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        final_top_k: int = 10,
    ) -> list[dict]:
        """
        Run vector + graph retrieval and return a ranked
        hybrid result set.

        LangSmith records this operation as the
        "hybrid-retrieval" child run when called from the
        traced RAG chain.
        """

        document_id = document_id.strip()
        query = query.strip()

        if not document_id:
            raise RetrievalError(
                "Document ID cannot be empty."
            )

        if not query:
            return []

        if final_top_k <= 0:
            return []

        effective_vector_top_k = (
            vector_top_k
            if vector_top_k is not None
            else self.vector_top_k
        )

        effective_graph_top_k = (
            graph_top_k
            if graph_top_k is not None
            else self.graph_top_k
        )

        try:

            # =================================================
            # VECTOR PATH
            # =================================================

            vector_results = (
                self.vector_retriever.retrieve(
                    document_id=document_id,
                    query=query,
                    top_k=effective_vector_top_k,
                )
            )

            # =================================================
            # GRAPH PATH
            # =================================================

            graph_results = (
                self.graph_retriever.retrieve(
                    query=query,
                    document_id=document_id,
                    top_k=effective_graph_top_k,
                )
            )

            # =================================================
            # NORMALIZE RESULT STRUCTURES
            # =================================================

            prepared_vector = (
                self._prepare_vector_results(
                    vector_results
                )
            )

            prepared_graph = (
                self._prepare_graph_results(
                    graph_results
                )
            )

            combined = (
                prepared_vector
                + prepared_graph
            )

            # =================================================
            # DEDUPLICATE
            # =================================================

            combined = (
                self._deduplicate(
                    combined
                )
            )

            # =================================================
            # RANK
            # =================================================

            combined.sort(
                key=lambda item: float(
                    item.get(
                        "hybrid_score",
                        0.0,
                    )
                ),
                reverse=True,
            )

            final_results = combined[
                :final_top_k
            ]

            logger.info(
                "Hybrid retrieval completed "
                "document_id=%s "
                "vector_results=%d "
                "graph_results=%d "
                "final_results=%d",
                document_id,
                len(vector_results),
                len(graph_results),
                len(final_results),
            )

            return final_results

        except RetrievalError:
            raise

        except Exception as exc:
            logger.exception(
                "Hybrid retrieval failed "
                "document_id=%s",
                document_id,
            )

            raise RetrievalError(
                "Hybrid retrieval failed."
            ) from exc

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def close(
        self,
    ) -> None:
        """
        Close resources owned by graph retrieval.
        """

        self.graph_retriever.close()