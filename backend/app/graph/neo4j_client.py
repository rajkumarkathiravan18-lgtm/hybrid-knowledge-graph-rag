from neo4j import GraphDatabase

from app.core.config import get_settings
from app.core.exceptions import GraphDatabaseError
from app.core.logging import get_logger
from app.models.graph import (
    Entity,
    Relationship,
    RelationshipType,
)


logger = get_logger(__name__)


class Neo4jClient:
    """
    Handles Neo4j connection, constraints,
    entity persistence, relationship persistence,
    document cleanup, graph queries,
    and driver lifecycle.
    """

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.neo4j_uri:
            raise GraphDatabaseError(
                "NEO4J_URI is not configured."
            )

        if not settings.neo4j_username:
            raise GraphDatabaseError(
                "NEO4J_USERNAME is not configured."
            )

        if not settings.neo4j_password:
            raise GraphDatabaseError(
                "NEO4J_PASSWORD is not configured."
            )

        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.database = settings.neo4j_database

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(
                    self.username,
                    settings.neo4j_password,
                ),
            )

        except Exception as exc:
            logger.exception(
                "Unable to create Neo4j driver"
            )

            raise GraphDatabaseError(
                "Unable to create Neo4j driver."
            ) from exc

    # =====================================================
    # CONNECTION
    # =====================================================

    def verify_connection(self) -> bool:
        """
        Verify Neo4j connectivity.
        """

        try:
            self.driver.verify_connectivity()

            logger.info(
                "Neo4j connectivity verified "
                "uri=%s database=%s",
                self.uri,
                self.database,
            )

            return True

        except Exception as exc:
            logger.exception(
                "Neo4j connectivity "
                "verification failed"
            )

            raise GraphDatabaseError(
                "Unable to connect to Neo4j."
            ) from exc

    # =====================================================
    # CONSTRAINTS
    # =====================================================

    def create_constraints(self) -> None:
        """
        Create uniqueness constraint for Entity nodes.
        """

        query = """
        CREATE CONSTRAINT entity_key_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.entity_key IS UNIQUE
        """

        try:
            self.driver.execute_query(
                query,
                database_=self.database,
            )

            logger.info(
                "Neo4j entity constraint verified"
            )

        except Exception as exc:
            logger.exception(
                "Unable to create Neo4j constraint"
            )

            raise GraphDatabaseError(
                "Unable to create Neo4j constraint."
            ) from exc

    # =====================================================
    # ENTITY KEY
    # =====================================================

    @staticmethod
    def _entity_key(
        entity: Entity,
    ) -> str:
        """
        Generate normalized entity identity.
        """

        normalized_name = " ".join(
            entity.name
            .casefold()
            .split()
        )

        return (
            f"{entity.type.value.casefold()}"
            f"::{normalized_name}"
        )

    @staticmethod
    def _entity_key_from_values(
        entity_name: str,
        entity_type: str,
    ) -> str:
        """
        Generate the same normalized entity identity
        using a name and entity type.

        This is used when resolving relationship
        source and target entities.
        """

        normalized_name = " ".join(
            entity_name
            .casefold()
            .split()
        )

        normalized_type = (
            entity_type
            .casefold()
            .strip()
        )

        return (
            f"{normalized_type}"
            f"::{normalized_name}"
        )

    # =====================================================
    # ENTITY PERSISTENCE
    # =====================================================

    def upsert_entities(
        self,
        entities: list[Entity],
    ) -> None:
        """
        Insert or update entities using MERGE.
        """

        if not entities:
            logger.info(
                "No entities supplied "
                "for persistence"
            )
            return

        rows = []

        for entity in entities:
            rows.append(
                {
                    "entity_key":
                        self._entity_key(
                            entity
                        ),

                    "name":
                        entity.name,

                    "entity_type":
                        entity.type.value,

                    "description":
                        entity.description,

                    "document_id":
                        entity.document_id,

                    "chunk_id":
                        entity.chunk_id,
                }
            )

        query = """
        UNWIND $entities AS item

        MERGE (e:Entity {
            entity_key: item.entity_key
        })

        ON CREATE SET
            e.name = item.name,
            e.entity_type = item.entity_type,
            e.description = item.description,
            e.created_at = datetime()

        SET
            e.last_document_id = item.document_id,
            e.last_chunk_id = item.chunk_id,
            e.updated_at = datetime()
        """

        try:
            self.driver.execute_query(
                query,
                entities=rows,
                database_=self.database,
            )

            logger.info(
                "Neo4j entities upserted "
                "count=%d",
                len(rows),
            )

        except Exception as exc:
            logger.exception(
                "Unable to persist "
                "Neo4j entities"
            )

            raise GraphDatabaseError(
                "Unable to persist "
                "graph entities."
            ) from exc

    # =====================================================
    # RELATIONSHIP PERSISTENCE
    # =====================================================

    def upsert_relationships(
        self,
        relationships: list[Relationship],
        entities: list[Entity],
    ) -> None:
        """
        Insert or update relationships between
        previously persisted Entity nodes.

        Relationship types are restricted to the
        controlled RelationshipType enum.

        Neo4j relationship types cannot be passed as
        ordinary Cypher parameters, so relationships
        are grouped by validated enum type before
        executing Cypher.
        """

        if not relationships:
            logger.info(
                "No relationships supplied "
                "for persistence"
            )
            return

        if not entities:
            logger.info(
                "No entities supplied for "
                "relationship resolution"
            )
            return

        # -------------------------------------------------
        # BUILD ENTITY LOOKUP
        # -------------------------------------------------

        entity_lookup: dict[
            str,
            list[Entity],
        ] = {}

        for entity in entities:
            normalized_name = " ".join(
                entity.name
                .casefold()
                .split()
            )

            entity_lookup.setdefault(
                normalized_name,
                [],
            ).append(entity)

        # -------------------------------------------------
        # GROUP RELATIONSHIPS BY VALIDATED TYPE
        # -------------------------------------------------

        grouped_rows: dict[
            str,
            list[dict],
        ] = {}

        allowed_relationship_types = {
            relationship_type.value
            for relationship_type
            in RelationshipType
        }

        skipped_relationships = 0

        for relationship in relationships:
            relationship_type = (
                relationship.relationship.value
            )

            if (
                relationship_type
                not in allowed_relationship_types
            ):
                skipped_relationships += 1
                continue

            source_name = " ".join(
                relationship.source
                .casefold()
                .split()
            )

            target_name = " ".join(
                relationship.target
                .casefold()
                .split()
            )

            source_candidates = (
                entity_lookup.get(
                    source_name,
                    [],
                )
            )

            target_candidates = (
                entity_lookup.get(
                    target_name,
                    [],
                )
            )

            if (
                not source_candidates
                or not target_candidates
            ):
                skipped_relationships += 1

                logger.warning(
                    "Skipping unresolved relationship "
                    "source=%s relationship=%s target=%s",
                    relationship.source,
                    relationship_type,
                    relationship.target,
                )

                continue

            # Entity extraction already uses controlled
            # names. If duplicate names with different
            # entity types exist, prefer candidates from
            # the same document/chunk as the relationship.

            source_entity = next(
                (
                    entity
                    for entity
                    in source_candidates
                    if (
                        entity.document_id
                        == relationship.document_id
                        and entity.chunk_id
                        == relationship.chunk_id
                    )
                ),
                source_candidates[0],
            )

            target_entity = next(
                (
                    entity
                    for entity
                    in target_candidates
                    if (
                        entity.document_id
                        == relationship.document_id
                        and entity.chunk_id
                        == relationship.chunk_id
                    )
                ),
                target_candidates[0],
            )

            source_key = self._entity_key(
                source_entity
            )

            target_key = self._entity_key(
                target_entity
            )

            if source_key == target_key:
                skipped_relationships += 1
                continue

            grouped_rows.setdefault(
                relationship_type,
                [],
            ).append(
                {
                    "source_key":
                        source_key,

                    "target_key":
                        target_key,

                    "evidence":
                        relationship.evidence,

                    "document_id":
                        relationship.document_id,

                    "chunk_id":
                        relationship.chunk_id,
                }
            )

        # -------------------------------------------------
        # WRITE EACH CONTROLLED RELATIONSHIP TYPE
        # -------------------------------------------------

        persisted_count = 0

        try:
            for (
                relationship_type,
                rows,
            ) in grouped_rows.items():

                if not rows:
                    continue

                # relationship_type is safe here because
                # it has already been validated against
                # RelationshipType enum values.

                query = f"""
                UNWIND $relationships AS item

                MATCH (source:Entity {{
                    entity_key: item.source_key
                }})

                MATCH (target:Entity {{
                    entity_key: item.target_key
                }})

                MERGE (source)-[r:{relationship_type} {{
                    document_id: item.document_id,
                    chunk_id: item.chunk_id
                }}]->(target)

                ON CREATE SET
                    r.created_at = datetime()

                SET
                    r.evidence = item.evidence,
                    r.updated_at = datetime()
                """

                self.driver.execute_query(
                    query,
                    relationships=rows,
                    database_=self.database,
                )

                persisted_count += len(rows)

            logger.info(
                "Neo4j relationships upserted "
                "count=%d skipped=%d",
                persisted_count,
                skipped_relationships,
            )

        except Exception as exc:
            logger.exception(
                "Unable to persist "
                "Neo4j relationships"
            )

            raise GraphDatabaseError(
                "Unable to persist "
                "graph relationships."
            ) from exc

    # =====================================================
    # DELETE DOCUMENT GRAPH DATA
    # =====================================================

    def delete_document_data(
        self,
        document_id: str,
    ) -> dict[str, int]:
        """
        Remove graph evidence belonging to one document.

        Strategy:

        1. Delete relationships whose document_id
           belongs to the document.

        2. Delete Entity nodes whose last_document_id
           matches the document only when those entities
           have no remaining relationships.

        Shared connected entities are preserved.
        """

        document_id = document_id.strip()

        if not document_id:
            raise GraphDatabaseError(
                "Document ID cannot be empty."
            )

        delete_relationships_query = """
        MATCH ()-[r]->()
        WHERE r.document_id = $document_id
        DELETE r
        RETURN count(r) AS deleted_relationships
        """

        delete_orphan_entities_query = """
        MATCH (e:Entity)
        WHERE e.last_document_id = $document_id
          AND NOT (e)--()
        WITH e
        DELETE e
        RETURN count(e) AS deleted_entities
        """

        try:
            relationship_records, _, _ = (
                self.driver.execute_query(
                    delete_relationships_query,
                    document_id=document_id,
                    database_=self.database,
                )
            )

            deleted_relationships = 0

            if relationship_records:
                deleted_relationships = int(
                    relationship_records[
                        0
                    ][
                        "deleted_relationships"
                    ]
                    or 0
                )

            entity_records, _, _ = (
                self.driver.execute_query(
                    delete_orphan_entities_query,
                    document_id=document_id,
                    database_=self.database,
                )
            )

            deleted_entities = 0

            if entity_records:
                deleted_entities = int(
                    entity_records[
                        0
                    ][
                        "deleted_entities"
                    ]
                    or 0
                )

            logger.info(
                "Neo4j document data deleted "
                "document_id=%s "
                "relationships=%d entities=%d",
                document_id,
                deleted_relationships,
                deleted_entities,
            )

            return {
                "deleted_relationships":
                    deleted_relationships,
                "deleted_entities":
                    deleted_entities,
            }

        except Exception as exc:
            logger.exception(
                "Unable to delete Neo4j "
                "document data document_id=%s",
                document_id,
            )

            raise GraphDatabaseError(
                "Unable to delete document "
                "graph data."
            ) from exc

    # =====================================================
    # GRAPH QUERY
    # =====================================================

    def query_entity(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Retrieve relationships connected to an entity.
        """

        entity_name = entity_name.strip()

        if not entity_name:
            return []

        if limit <= 0:
            return []

        query = """
        MATCH (source:Entity)-[r]->(target:Entity)

        WHERE toLower(source.name)
            CONTAINS toLower($entity_name)
           OR
              toLower(target.name)
            CONTAINS toLower($entity_name)

        RETURN
            source.name AS source,
            source.entity_type AS source_type,
            type(r) AS relationship,
            target.name AS target,
            target.entity_type AS target_type,
            r.evidence AS evidence,
            r.document_id AS document_id,
            r.chunk_id AS chunk_id

        LIMIT $limit
        """

        try:
            records, _, _ = (
                self.driver.execute_query(
                    query,
                    entity_name=entity_name,
                    limit=limit,
                    database_=self.database,
                    routing_="r",
                )
            )

            return [
                record.data()
                for record in records
            ]

        except Exception as exc:
            logger.exception(
                "Neo4j entity query failed "
                "entity_name=%s",
                entity_name,
            )

            raise GraphDatabaseError(
                "Unable to query graph."
            ) from exc

    # =====================================================
    # DRIVER LIFECYCLE
    # =====================================================

    def close(self) -> None:
        """
        Close Neo4j driver connection pool.
        """

        if self.driver:
            self.driver.close()

            logger.info(
                "Neo4j driver closed"
            )