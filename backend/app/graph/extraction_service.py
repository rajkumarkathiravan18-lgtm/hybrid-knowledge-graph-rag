from app.core.logging import get_logger
from app.graph.entity_extractor import EntityExtractor
from app.graph.relationship_extractor import RelationshipExtractor
from app.models.document import DocumentChunk
from app.models.graph import GraphExtractionResult


logger = get_logger(__name__)


class GraphExtractionService:
    """
    Orchestrates knowledge graph extraction.

    Pipeline:

        DocumentChunk
            ↓
        EntityExtractor
            ↓
        RelationshipExtractor
            ↓
        GraphExtractionResult
    """

    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()

    def extract_chunk(
        self,
        chunk: DocumentChunk,
    ) -> GraphExtractionResult:
        """
        Extract entities and relationships from one chunk.
        """

        logger.info(
            "Starting graph extraction "
            "document_id=%s chunk_id=%s",
            chunk.document_id,
            chunk.chunk_id,
        )

        entities = self.entity_extractor.extract(
            chunk
        )

        relationships = (
            self.relationship_extractor.extract(
                chunk,
                entities,
            )
        )

        result = GraphExtractionResult(
            entities=entities,
            relationships=relationships,
        )

        logger.info(
            "Graph extraction completed "
            "document_id=%s chunk_id=%s "
            "entities=%d relationships=%d",
            chunk.document_id,
            chunk.chunk_id,
            len(entities),
            len(relationships),
        )

        return result

    def extract_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> GraphExtractionResult:
        """
        Extract graph information from multiple chunks.
        """

        all_entities = []
        all_relationships = []

        for chunk in chunks:
            result = self.extract_chunk(
                chunk
            )

            all_entities.extend(
                result.entities
            )

            all_relationships.extend(
                result.relationships
            )

        logger.info(
            "Multi-chunk graph extraction completed "
            "chunks=%d entities=%d relationships=%d",
            len(chunks),
            len(all_entities),
            len(all_relationships),
        )

        return GraphExtractionResult(
            entities=all_entities,
            relationships=all_relationships,
        )