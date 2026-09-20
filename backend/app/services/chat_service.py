from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from app.core.config import get_settings
from app.core.exceptions import (
    GuardrailBlockedError,
    GuardrailError,
    LLMServiceError,
)
from app.core.logging import get_logger
from app.guardrails.service import GuardrailService
from app.llm.chain import RAGChain
from app.memory.conversation_store import ConversationStore


logger = get_logger(__name__)


class ChatService:
    """
    Application-level chat orchestration service.

    Pipeline:

        User Question
            |
            v
        Input Guardrail
            |
            v
        Resolve Conversation
            |
            v
        Load Persistent History
            |
            v
        Hybrid RAG
            |
            +--> Document evidence
            |
            +--> Controlled web fallback
            |
            v
        Output Guardrail
            |
            v
        Save User + Assistant Messages
            |
            v
        Final Response

    Conversation history is persisted in SQLite and converted
    into LangChain messages before being supplied to RAGChain.

    The source type returned by RAGChain is preserved through
    memory and the API response.

    Supported source types:

        document
        web
        document+web
        insufficient
    """

    VALID_SOURCE_TYPES = {
        "document",
        "web",
        "document+web",
        "insufficient",
    }

    def __init__(self) -> None:
        try:
            self.settings = get_settings()

            self.guardrails = GuardrailService()

            self.rag_chain = RAGChain()

            self.memory_store = ConversationStore(
                database_path=self.settings.memory_db_path,
            )

            logger.info(
                "ChatService initialized successfully"
            )

        except Exception as exc:
            logger.exception(
                "Unable to initialize ChatService"
            )

            raise LLMServiceError(
                "Unable to initialize chat service."
            ) from exc

    # ========================================================
    # CONVERSATION RESOLUTION
    # ========================================================

    def _resolve_conversation(
        self,
        document_id: str,
        question: str,
        conversation_id: str | None,
    ) -> str:
        """
        Return an existing conversation ID or create a new one.

        A conversation is associated with the document used
        when it is created.
        """

        if conversation_id:
            conversation = (
                self.memory_store.get_conversation(
                    conversation_id
                )
            )

            if conversation is None:
                raise ValueError(
                    "Conversation does not exist: "
                    f"{conversation_id}"
                )

            stored_document_id = conversation.get(
                "document_id"
            )

            if (
                stored_document_id
                and stored_document_id != document_id
            ):
                raise ValueError(
                    "Conversation belongs to a different "
                    "document."
                )

            if not stored_document_id:
                self.memory_store.update_document(
                    conversation_id=conversation_id,
                    document_id=document_id,
                )

            return conversation_id

        conversation = (
            self.memory_store.create_conversation(
                document_id=document_id,
                first_message=question,
            )
        )

        return str(
            conversation["conversation_id"]
        )

    # ========================================================
    # HISTORY
    # ========================================================

    def _load_history(
        self,
        conversation_id: str,
    ) -> list[BaseMessage]:
        """
        Load recent persistent conversation messages and
        convert them into LangChain message objects.

        System messages stored in SQLite are intentionally not
        passed through here. The authoritative system prompt is
        controlled by RAGChain.
        """

        stored_messages = (
            self.memory_store.get_messages(
                conversation_id=conversation_id,
                limit=self.settings.memory_history_limit,
            )
        )

        history: list[BaseMessage] = []

        for message in stored_messages:
            role = message.get("role")

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            if role == "user":
                history.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":
                history.append(
                    AIMessage(
                        content=content
                    )
                )

        return history

    # ========================================================
    # SOURCE TYPE
    # ========================================================

    def _resolve_source_type(
        self,
        rag_result: dict,
    ) -> str:
        """
        Validate the source type returned by RAGChain.

        Unknown values are converted to "insufficient"
        rather than allowing unsupported provenance labels
        to enter persistent memory or API responses.
        """

        source_type = str(
            rag_result.get(
                "source_type",
                "insufficient",
            )
        ).strip().lower()

        if source_type not in self.VALID_SOURCE_TYPES:
            logger.warning(
                "Unexpected RAG source_type=%r. "
                "Using insufficient.",
                source_type,
            )

            return "insufficient"

        return source_type

    # ========================================================
    # CHAT
    # ========================================================

    async def chat(
        self,
        document_id: str,
        question: str,
        conversation_id: str | None = None,
    ) -> dict:
        """
        Process one complete secured RAG request.

        Returns:

            {
                "answer": str,
                "conversation_id": str,
                "source_type": str,
                "retrieval_results": list,
                "web_results": list,
                "context": str,
                "document_context": str,
                "web_context": str,
                "document_sufficient": bool,
                "sufficiency_reason": str,
            }
        """

        document_id = document_id.strip()
        question = question.strip()

        if not document_id:
            raise ValueError(
                "Document ID cannot be empty."
            )

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # ====================================================
        # STEP 1 - INPUT GUARDRAIL
        # ====================================================

        logger.info(
            "Running input guardrail "
            "document_id=%s",
            document_id,
        )

        try:
            input_allowed = (
                await self.guardrails.check_input(
                    question
                )
            )

        except GuardrailError:
            logger.exception(
                "Input guardrail execution failed"
            )
            raise

        if not input_allowed:
            logger.warning(
                "Input blocked by guardrail "
                "document_id=%s",
                document_id,
            )

            raise GuardrailBlockedError(
                "The request was blocked by "
                "the input security guardrail."
            )

        # ====================================================
        # STEP 2 - RESOLVE CONVERSATION
        # ====================================================

        resolved_conversation_id = (
            self._resolve_conversation(
                document_id=document_id,
                question=question,
                conversation_id=conversation_id,
            )
        )

        logger.info(
            "Conversation resolved "
            "conversation_id=%s "
            "document_id=%s",
            resolved_conversation_id,
            document_id,
        )

        # ====================================================
        # STEP 3 - LOAD PERSISTENT HISTORY
        # ====================================================

        history = self._load_history(
            resolved_conversation_id
        )

        logger.info(
            "Loaded conversation history "
            "conversation_id=%s "
            "message_count=%s",
            resolved_conversation_id,
            len(history),
        )

        # ====================================================
        # STEP 4 - HYBRID RAG + CONTROLLED WEB FALLBACK
        # ====================================================

        logger.info(
            "Running RAG pipeline "
            "document_id=%s "
            "conversation_id=%s",
            document_id,
            resolved_conversation_id,
        )

        try:
            rag_result = self.rag_chain.invoke(
                document_id=document_id,
                question=question,
                history=history,
            )

        except Exception as exc:
            logger.exception(
                "RAG pipeline failed "
                "document_id=%s "
                "conversation_id=%s",
                document_id,
                resolved_conversation_id,
            )

            if isinstance(
                exc,
                LLMServiceError,
            ):
                raise

            raise LLMServiceError(
                "RAG generation failed."
            ) from exc

        answer = str(
            rag_result.get(
                "answer",
                "",
            )
        ).strip()

        if not answer:
            raise LLMServiceError(
                "The language model returned "
                "an empty response."
            )

        source_type = self._resolve_source_type(
            rag_result
        )

        logger.info(
            "RAG source resolved "
            "document_id=%s "
            "conversation_id=%s "
            "source_type=%s",
            document_id,
            resolved_conversation_id,
            source_type,
        )

        # ====================================================
        # STEP 5 - OUTPUT GUARDRAIL
        # ====================================================

        logger.info(
            "Running output guardrail "
            "document_id=%s "
            "conversation_id=%s",
            document_id,
            resolved_conversation_id,
        )

        try:
            output_allowed = (
                await self.guardrails.check_output(
                    answer
                )
            )

        except GuardrailError:
            logger.exception(
                "Output guardrail execution failed"
            )
            raise

        if not output_allowed:
            logger.warning(
                "Output blocked by guardrail "
                "document_id=%s "
                "conversation_id=%s",
                document_id,
                resolved_conversation_id,
            )

            raise GuardrailBlockedError(
                "The generated response was blocked "
                "by the output security guardrail."
            )

        # ====================================================
        # STEP 6 - SAVE CONVERSATION MEMORY
        # ====================================================

        try:
            self.memory_store.add_message(
                conversation_id=(
                    resolved_conversation_id
                ),
                role="user",
                content=question,
            )

            self.memory_store.add_message(
                conversation_id=(
                    resolved_conversation_id
                ),
                role="assistant",
                content=answer,
                source_type=source_type,
            )

        except Exception as exc:
            logger.exception(
                "Unable to persist conversation "
                "conversation_id=%s",
                resolved_conversation_id,
            )

            raise LLMServiceError(
                "The answer was generated but "
                "conversation memory could not "
                "be persisted."
            ) from exc

        # ====================================================
        # STEP 7 - FINAL RESPONSE
        # ====================================================

        logger.info(
            "Chat request completed "
            "document_id=%s "
            "conversation_id=%s "
            "source_type=%s",
            document_id,
            resolved_conversation_id,
            source_type,
        )

        return {
            "answer":
                answer,

            "conversation_id":
                resolved_conversation_id,

            "source_type":
                source_type,

            "retrieval_results":
                rag_result.get(
                    "retrieval_results",
                    [],
                ),

            "web_results":
                rag_result.get(
                    "web_results",
                    [],
                ),

            "context":
                rag_result.get(
                    "context",
                    "",
                ),

            "document_context":
                rag_result.get(
                    "document_context",
                    "",
                ),

            "web_context":
                rag_result.get(
                    "web_context",
                    "",
                ),

            "document_sufficient":
                bool(
                    rag_result.get(
                        "document_sufficient",
                        False,
                    )
                ),

            "sufficiency_reason":
                str(
                    rag_result.get(
                        "sufficiency_reason",
                        "",
                    )
                ),
        }

    # ========================================================
    # CONVERSATION ACCESS
    # ========================================================

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict | None:
        """
        Retrieve a complete persisted conversation.
        """

        return self.memory_store.get_conversation(
            conversation_id
        )

    def list_conversations(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return recent persisted conversations.
        """

        return self.memory_store.list_conversations(
            limit=limit
        )

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """
        Delete a conversation and all associated messages.
        """

        return self.memory_store.delete_conversation(
            conversation_id
        )

    # ========================================================
    # RESOURCE CLEANUP
    # ========================================================

    def close(self) -> None:
        """
        Release resources used by the RAG pipeline.
        """

        try:
            self.rag_chain.close()

        except Exception:
            logger.exception(
                "Error while closing ChatService"
            )