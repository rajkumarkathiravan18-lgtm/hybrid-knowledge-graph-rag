from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(str, Enum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    CONCEPT = "Concept"
    PRODUCT = "Product"
    TECHNOLOGY = "Technology"
    EVENT = "Event"
    DOCUMENT = "Document"


class RelationshipType(str, Enum):
    USES = "USES"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CREATED_BY = "CREATED_BY"
    LOCATED_IN = "LOCATED_IN"
    INTEGRATES_WITH = "INTEGRATES_WITH"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    MENTIONS = "MENTIONS"


class Entity(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    type: EntityType

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    document_id: str

    chunk_id: str

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = " ".join(value.split())

        if not value:
            raise ValueError(
                "Entity name cannot be empty."
            )

        return value


class ExtractedEntity(BaseModel):
    """
    LLM-facing entity model.

    document_id/chunk_id are intentionally NOT generated
    by the LLM. They are attached by application code.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    type: EntityType

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return " ".join(value.split())


class EntityExtractionPayload(BaseModel):
    entities: list[ExtractedEntity] = Field(
        default_factory=list
    )


class Relationship(BaseModel):
    source: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    relationship: RelationshipType

    target: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    evidence: str | None = Field(
        default=None,
        max_length=3000,
    )

    document_id: str

    chunk_id: str

    @field_validator(
        "source",
        "target",
    )
    @classmethod
    def normalize_entity_name(
        cls,
        value: str,
    ) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_relationship(self) -> "Relationship":
        if (
            self.source.casefold()
            == self.target.casefold()
        ):
            raise ValueError(
                "Self-referencing relationships are not allowed."
            )

        return self


class ExtractedRelationship(BaseModel):
    """
    LLM-facing relationship representation.
    """

    source: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    relationship: RelationshipType

    target: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    evidence: str | None = Field(
        default=None,
        max_length=3000,
    )


class RelationshipExtractionPayload(BaseModel):
    relationships: list[
        ExtractedRelationship
    ] = Field(
        default_factory=list
    )


class GraphExtractionResult(BaseModel):
    entities: list[Entity] = Field(
        default_factory=list
    )

    relationships: list[Relationship] = Field(
        default_factory=list
    )


class GraphQueryResult(BaseModel):
    source: str

    source_type: str

    relationship: str

    target: str

    target_type: str

    evidence: str | None = None

    document_id: str | None = None

    chunk_id: str | None = None