from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.graph.neo4j_client import Neo4jClient
from app.observability.langsmith import traced


logger = get_logger(__name__)


class GraphRetriever:
    """
    High-level Neo4j graph retrieval service.

    Responsibilities:
    - receive a natural-language query
    - find graph entities mentioned in that query
    - retrieve relationships connected to those entities
    - preserve graph evidence and provenance metadata
    """

    def __init__(
        self,
        client: Neo4jClient | None = None,
    ) -> None:
        settings = get_settings()

        self.client = client or Neo4jClient()
        self.default_top_k = settings.graph_top_k

    # =====================================================
    # ENTITY DISCOVERY
    # =====================================================

    def find_query_entities(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find known graph entities whose names occur
        inside the user's natural-language question.

        Example:

            Query:
                "How does Neo4j work with FAISS?"

            Matches:
                Neo4j
                FAISS

        When document_id is provided, entities are restricted
        to those associated with that document where possible.
        """

        query = query.strip()

        if not query:
            return []

        if limit <= 0:
            return []

        cypher = """
        MATCH (entity:Entity)

        WHERE toLower($query)
              CONTAINS toLower(entity.name)

          AND (
                $document_id IS NULL

                OR entity.last_document_id = $document_id

                OR EXISTS {
                    MATCH (entity)-[r]-()
                    WHERE r.document_id = $document_id
                }
          )

        RETURN DISTINCT
            entity.name AS name,
            entity.entity_type AS entity_type,
            entity.description AS description

        ORDER BY size(entity.name) DESC

        LIMIT $limit
        """

        try:
            records, _, _ = (
                self.client.driver.execute_query(
                    cypher,
                    query=query,
                    document_id=document_id,
                    limit=limit,
                    database_=self.client.database,
                    routing_="r",
                )
            )

            return [
                record.data()
                for record in records
            ]

        except Exception as exc:
            logger.exception(
                "Graph entity discovery failed"
            )

            raise RetrievalError(
                "Graph entity discovery failed."
            ) from exc

    # =====================================================
    # RETRIEVAL
    # =====================================================

    @traced(
        name="graph-retrieval",
        run_type="retriever",
    )
    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve graph evidence relevant to the query.

        This operation is traced separately in LangSmith so
        Neo4j graph retrieval can be inspected independently
        inside the Hybrid RAG retrieval trace.
        """

        query = query.strip()

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
            entities = self.find_query_entities(
                query=query,
                document_id=document_id,
                limit=effective_top_k,
            )

            if not entities:
                logger.info(
                    "No graph entities found for query"
                )
                return []

            results = []
            seen = set()

            for entity in entities:
                entity_name = entity["name"]

                relationships = (
                    self.client.query_entity(
                        entity_name=entity_name,
                        limit=effective_top_k,
                    )
                )

                for relationship in relationships:

                    # Keep graph evidence scoped to the
                    # active document when requested.
                    if document_id:
                        relationship_document_id = (
                            relationship.get(
                                "document_id"
                            )
                        )

                        if (
                            relationship_document_id
                            != document_id
                        ):
                            continue

                    dedupe_key = (
                        relationship.get("source"),
                        relationship.get(
                            "relationship"
                        ),
                        relationship.get("target"),
                        relationship.get(
                            "document_id"
                        ),
                        relationship.get(
                            "chunk_id"
                        ),
                    )

                    if dedupe_key in seen:
                        continue

                    seen.add(dedupe_key)

                    results.append(
                        {
                            "retrieval_type":
                                "graph",

                            "matched_entity":
                                entity_name,

                            "source":
                                relationship.get(
                                    "source"
                                ),

                            "source_type":
                                relationship.get(
                                    "source_type"
                                ),

                            "relationship":
                                relationship.get(
                                    "relationship"
                                ),

                            "target":
                                relationship.get(
                                    "target"
                                ),

                            "target_type":
                                relationship.get(
                                    "target_type"
                                ),

                            "evidence":
                                relationship.get(
                                    "evidence"
                                ),

                            "document_id":
                                relationship.get(
                                    "document_id"
                                ),

                            "chunk_id":
                                relationship.get(
                                    "chunk_id"
                                ),
                        }
                    )

                    if (
                        len(results)
                        >= effective_top_k
                    ):
                        break

                if len(results) >= effective_top_k:
                    break

            logger.info(
                "Graph retrieval completed "
                "entities=%d results=%d",
                len(entities),
                len(results),
            )

            return results

        except RetrievalError:
            raise

        except Exception as exc:
            logger.exception(
                "Graph retrieval failed"
            )

            raise RetrievalError(
                "Graph retrieval failed."
            ) from exc

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def close(self) -> None:
        """
        Close the underlying Neo4j client.
        """

        self.client.close()