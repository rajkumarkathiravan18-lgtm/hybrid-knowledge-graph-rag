from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.llm.prompts import get_rag_system_prompt
from app.observability.langsmith import traced
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.evidence_sufficiency import (
    EvidenceStatus,
    EvidenceSufficiencyService,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.web.context_builder import WebContextBuilder
from app.web.evidence_validator import WebEvidenceValidator
from app.web.search_service import WebSearchService


logger = get_logger(__name__)


class RAGChain:
    """
    Main Hybrid Knowledge Graph RAG generation pipeline.

    Flow:

        Original User Question
                |
                v
        Conversation History
                |
                v
        QueryRewriter
                |
                v
        Standalone Retrieval Query
                |
                v
        HybridRetriever
        (FAISS + Neo4j)
                |
                v
        Document ContextBuilder
                |
                v
        EvidenceSufficiencyService
                |
        +-----------+-----------+---------------+
        |                       |               |
        | sufficient            | partial       | insufficient
        v                       v               v
     Document                Web Search       Web Search
       only                     |               |
                                v               v
                        WebEvidenceValidator
                                |
                        +-------+-------+
                        |               |
                     relevant       irrelevant
                        |               |
                        v               v
                  Document+Web     Document only /
                                   Insufficient

    Source types:

        document
        web
        document+web
        insufficient

    Important rules:

    - The rewritten query is used for retrieval,
      sufficiency classification, and web search.

    - The original user question is preserved for final
      generation and conversational UX.

    - Web evidence is treated as untrusted external data.

    - Web results are not accepted merely because the
      search engine returned results. They must pass
      semantic relevance validation.

    - Combined document + web context is globally bounded
      by MAX_CONTEXT_CHARS.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        llm_client: LLMClient | None = None,
        sufficiency_service: EvidenceSufficiencyService | None = None,
        query_rewriter: QueryRewriter | None = None,
        web_search_service: WebSearchService | None = None,
        web_context_builder: WebContextBuilder | None = None,
        web_evidence_validator: WebEvidenceValidator | None = None,
    ) -> None:
        settings = get_settings()

        self.max_context_chars = (
            settings.max_context_chars
        )

        if self.max_context_chars <= 0:
            raise ValueError(
                "MAX_CONTEXT_CHARS must be greater than 0."
            )

        self.retriever = (
            retriever
            or HybridRetriever()
        )

        self.context_builder = (
            context_builder
            or ContextBuilder()
        )

        self.llm_client = (
            llm_client
            or LLMClient()
        )

        self.model = (
            self.llm_client.get_model()
        )

        self.sufficiency_service = (
            sufficiency_service
            or EvidenceSufficiencyService(
                llm_client=self.llm_client
            )
        )

        self.query_rewriter = (
            query_rewriter
            or QueryRewriter(
                llm_client=self.llm_client
            )
        )

        self.web_search_service = (
            web_search_service
            or WebSearchService()
        )

        self.web_context_builder = (
            web_context_builder
            or WebContextBuilder()
        )

        self.web_evidence_validator = (
            web_evidence_validator
            or WebEvidenceValidator(
                llm_client=self.llm_client
            )
        )

    # =====================================================
    # HISTORY VALIDATION
    # =====================================================

    @staticmethod
    def _prepare_history(
        history: Sequence[BaseMessage] | None,
    ) -> list[BaseMessage]:
        """
        Keep only user and assistant conversation messages.

        Stored system messages are intentionally ignored
        because the application owns the system instruction.
        """

        if not history:
            return []

        prepared: list[BaseMessage] = []

        for message in history:
            if isinstance(
                message,
                (
                    HumanMessage,
                    AIMessage,
                ),
            ):
                prepared.append(
                    message
                )

        return prepared

    # =====================================================
    # DOCUMENT EVIDENCE CHECK
    # =====================================================

    @staticmethod
    def _has_document_evidence(
        retrieval_results: list[dict],
    ) -> bool:
        """
        Return True only when retrieval produced at least
        one usable vector or graph evidence item.
        """

        for result in retrieval_results:
            retrieval_type = result.get(
                "retrieval_type"
            )

            if retrieval_type not in {
                "vector",
                "graph",
            }:
                continue

            text = str(
                result.get("text")
                or result.get("graph_evidence")
                or ""
            ).strip()

            if text:
                return True

        return False

    # =====================================================
    # CONTEXT BOUNDING
    # =====================================================

    @staticmethod
    def _truncate_context(
        context: str,
        max_chars: int,
    ) -> str:
        """
        Safely truncate a context section to a fixed
        character budget.
        """

        context = context.strip()

        if not context:
            return ""

        if max_chars <= 0:
            return ""

        if len(context) <= max_chars:
            return context

        marker = (
            "\n[TRUNCATED DUE TO GLOBAL CONTEXT LIMIT]"
        )

        if max_chars <= len(marker):
            return context[:max_chars]

        available = (
            max_chars
            - len(marker)
        )

        return (
            context[:available].rstrip()
            + marker
        )

    def _bound_single_context(
        self,
        context: str,
    ) -> str:
        """
        Apply the global context limit to one evidence
        source.
        """

        return self._truncate_context(
            context=context,
            max_chars=self.max_context_chars,
        )

    # =====================================================
    # DOCUMENT + WEB CONTEXT
    # =====================================================

    def _combine_contexts(
        self,
        document_context: str,
        web_context: str,
    ) -> str:
        """
        Combine document and web evidence while keeping
        their provenance explicitly separated.

        The final combined string is guaranteed not to
        exceed MAX_CONTEXT_CHARS.

        When both evidence classes are available, the
        available evidence budget is split approximately
        50/50 so neither source can consume the complete
        generation context.
        """

        document_context = (
            document_context.strip()
        )

        web_context = (
            web_context.strip()
        )

        if not document_context:
            return self._bound_single_context(
                web_context
            )

        if not web_context:
            return self._bound_single_context(
                document_context
            )

        document_header = (
            "DOCUMENT EVIDENCE\n"
            "=================\n\n"
        )

        web_header = (
            "WEB EVIDENCE\n"
            "============\n\n"
        )

        divider = (
            "\n\n"
            "----------------------------------------"
            "\n\n"
        )

        structural_chars = (
            len(document_header)
            + len(web_header)
            + len(divider)
        )

        evidence_budget = (
            self.max_context_chars
            - structural_chars
        )

        if evidence_budget <= 0:
            logger.warning(
                "MAX_CONTEXT_CHARS=%d is too small "
                "for mixed evidence headers.",
                self.max_context_chars,
            )

            raw_context = (
                document_header
                + document_context
                + divider
                + web_header
                + web_context
            )

            return self._truncate_context(
                context=raw_context,
                max_chars=self.max_context_chars,
            )

        document_budget = (
            evidence_budget // 2
        )

        web_budget = (
            evidence_budget
            - document_budget
        )

        bounded_document = (
            self._truncate_context(
                context=document_context,
                max_chars=document_budget,
            )
        )

        bounded_web = (
            self._truncate_context(
                context=web_context,
                max_chars=web_budget,
            )
        )

        combined_context = (
            document_header
            + bounded_document
            + divider
            + web_header
            + bounded_web
        )

        # Defensive final enforcement.
        if (
            len(combined_context)
            > self.max_context_chars
        ):
            combined_context = (
                self._truncate_context(
                    context=combined_context,
                    max_chars=self.max_context_chars,
                )
            )

        logger.info(
            "Mixed context built "
            "document_chars=%d "
            "web_chars=%d "
            "combined_chars=%d "
            "limit=%d",
            len(bounded_document),
            len(bounded_web),
            len(combined_context),
            self.max_context_chars,
        )

        return combined_context

    # =====================================================
    # MESSAGE CONSTRUCTION
    # =====================================================

    def build_messages(
        self,
        question: str,
        context: str,
        history: Sequence[BaseMessage] | None = None,
        source_type: str = "document",
    ) -> list[BaseMessage]:
        """
        Construct the LangChain message sequence.

        The original user question is supplied here.

        The rewritten retrieval query is intentionally not
        substituted for the user's conversational message.
        """

        question = question.strip()

        context = (
            self._bound_single_context(
                context
            )
        )

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        system_prompt = (
            get_rag_system_prompt()
        )

        messages: list[BaseMessage] = [
            SystemMessage(
                content=system_prompt
            )
        ]

        messages.extend(
            self._prepare_history(
                history
            )
        )

        current_message = f"""
EVIDENCE SOURCE TYPE
====================

{source_type}

RETRIEVED EVIDENCE
==================

{context}

==================
END RETRIEVED EVIDENCE

USER QUESTION
=============

{question}
""".strip()

        messages.append(
            HumanMessage(
                content=current_message
            )
        )

        return messages

    # =====================================================
    # MODEL GENERATION
    # =====================================================

    def _generate_answer(
        self,
        question: str,
        context: str,
        history: Sequence[BaseMessage] | None,
        source_type: str,
    ) -> str:
        """
        Generate the final grounded answer using the
        original user question.
        """

        messages = self.build_messages(
            question=question,
            context=context,
            history=history,
            source_type=source_type,
        )

        response = self.model.invoke(
            messages
        )

        answer = response.content

        if not isinstance(
            answer,
            str,
        ):
            answer = str(
                answer
            )

        answer = answer.strip()

        if not answer:
            raise LLMServiceError(
                "Language model returned "
                "an empty answer."
            )

        return answer

    # =====================================================
    # INVOCATION
    # =====================================================

    @traced(
        name="hybrid-rag-generation",
        run_type="chain",
    )
    def invoke(
        self,
        document_id: str,
        question: str,
        history: Sequence[BaseMessage] | None = None,
    ) -> dict:
        """
        Execute conversation-aware document-first RAG with
        controlled three-way evidence routing.

        Routing:

            sufficient
                -> document only

            partial
                -> web search
                -> validate web
                -> document + web when relevant
                -> document only when web is rejected

            insufficient
                -> web search
                -> validate web
                -> web only when relevant
                -> insufficient when web is rejected
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

        try:
            # =================================================
            # 1. PREPARE CONVERSATION HISTORY
            # =================================================

            prepared_history = (
                self._prepare_history(
                    history
                )
            )

            # =================================================
            # 2. CONVERSATION-AWARE QUERY REWRITE
            # =================================================

            rewrite_result = (
                self.query_rewriter.rewrite(
                    question=question,
                    history=prepared_history,
                )
            )

            retrieval_query = (
                rewrite_result.query.strip()
            )

            if not retrieval_query:
                retrieval_query = question

            query_was_rewritten = (
                rewrite_result.rewritten
            )

            logger.info(
                "Retrieval query prepared "
                "document_id=%s "
                "rewritten=%s "
                "original_question=%r "
                "retrieval_query=%r",
                document_id,
                query_was_rewritten,
                question,
                retrieval_query,
            )

            # =================================================
            # 3. DOCUMENT HYBRID RETRIEVAL
            # =================================================

            retrieval_results = (
                self.retriever.retrieve(
                    document_id=document_id,
                    query=retrieval_query,
                )
            )

            # =================================================
            # 4. DOCUMENT CONTEXT
            # =================================================

            document_context = (
                self.context_builder.build(
                    retrieval_results
                )
            )

            document_context = (
                self._bound_single_context(
                    document_context
                )
            )

            has_document_evidence = (
                self._has_document_evidence(
                    retrieval_results
                )
            )

            # =================================================
            # 5. DOCUMENT EVIDENCE CLASSIFICATION
            # =================================================

            if has_document_evidence:
                sufficiency_decision = (
                    self.sufficiency_service.evaluate(
                        question=retrieval_query,
                        context=document_context,
                    )
                )

                evidence_status = (
                    sufficiency_decision.status
                )

                sufficiency_reason = (
                    sufficiency_decision.reason
                )

            else:
                evidence_status = (
                    EvidenceStatus.INSUFFICIENT
                )

                sufficiency_reason = (
                    "No usable document evidence "
                    "was retrieved."
                )

            document_sufficient = (
                evidence_status
                == EvidenceStatus.SUFFICIENT
            )

            logger.info(
                "Document evidence evaluated "
                "document_id=%s "
                "status=%s "
                "reason=%s",
                document_id,
                evidence_status.value,
                sufficiency_reason,
            )

            # =================================================
            # 6. SUFFICIENT -> DOCUMENT ONLY
            # =================================================

            if document_sufficient:
                source_type = "document"

                answer = self._generate_answer(
                    question=question,
                    context=document_context,
                    history=prepared_history,
                    source_type=source_type,
                )

                logger.info(
                    "RAG invocation completed "
                    "document_id=%s "
                    "source_type=%s "
                    "retrieval_results=%d "
                    "query_rewritten=%s",
                    document_id,
                    source_type,
                    len(retrieval_results),
                    query_was_rewritten,
                )

                return {
                    "answer": answer,
                    "source_type": source_type,
                    "retrieval_results":
                        retrieval_results,
                    "web_results": [],
                    "context":
                        document_context,
                    "document_context":
                        document_context,
                    "web_context": "",
                    "document_sufficient":
                        True,
                    "evidence_status":
                        evidence_status.value,
                    "sufficiency_reason":
                        sufficiency_reason,
                    "web_evidence_relevant":
                        False,
                    "web_validation_reason":
                        "Web search was not required.",
                    "original_question":
                        question,
                    "retrieval_query":
                        retrieval_query,
                    "query_rewritten":
                        query_was_rewritten,
                }

            # =================================================
            # 7. PARTIAL / INSUFFICIENT -> WEB SEARCH
            # =================================================

            logger.info(
                "Web supplementation activated "
                "document_id=%s "
                "evidence_status=%s "
                "retrieval_query=%r",
                document_id,
                evidence_status.value,
                retrieval_query,
            )

            web_results = (
                self.web_search_service.search(
                    query=retrieval_query
                )
            )

            web_context = (
                self.web_context_builder.build(
                    web_results
                )
            )

            web_context = (
                self._bound_single_context(
                    web_context
                )
            )

            # =================================================
            # 8. VALIDATE WEB EVIDENCE
            # =================================================

            web_validation_reason = (
                "No web evidence was retrieved."
            )

            if web_results and web_context.strip():
                web_validation = (
                    self.web_evidence_validator.evaluate(
                        question=retrieval_query,
                        context=web_context,
                    )
                )

                has_web_evidence = (
                    web_validation.relevant
                )

                web_validation_reason = (
                    web_validation.reason
                )

            else:
                has_web_evidence = False

            logger.info(
                "Web evidence evaluated "
                "document_id=%s "
                "relevant=%s "
                "reason=%s",
                document_id,
                has_web_evidence,
                web_validation_reason,
            )

            # =================================================
            # 9. PARTIAL + RELEVANT WEB -> DOCUMENT + WEB
            # =================================================

            if (
                evidence_status
                == EvidenceStatus.PARTIAL
                and has_web_evidence
            ):
                source_type = "document+web"

                combined_context = (
                    self._combine_contexts(
                        document_context=
                            document_context,
                        web_context=
                            web_context,
                    )
                )

                answer = self._generate_answer(
                    question=question,
                    context=combined_context,
                    history=prepared_history,
                    source_type=source_type,
                )

                logger.info(
                    "RAG invocation completed "
                    "document_id=%s "
                    "source_type=%s "
                    "retrieval_results=%d "
                    "web_results=%d "
                    "context_chars=%d "
                    "query_rewritten=%s",
                    document_id,
                    source_type,
                    len(retrieval_results),
                    len(web_results),
                    len(combined_context),
                    query_was_rewritten,
                )

                return {
                    "answer": answer,
                    "source_type": source_type,
                    "retrieval_results":
                        retrieval_results,
                    "web_results":
                        web_results,
                    "context":
                        combined_context,
                    "document_context":
                        document_context,
                    "web_context":
                        web_context,
                    "document_sufficient":
                        False,
                    "evidence_status":
                        evidence_status.value,
                    "sufficiency_reason":
                        sufficiency_reason,
                    "web_evidence_relevant":
                        True,
                    "web_validation_reason":
                        web_validation_reason,
                    "original_question":
                        question,
                    "retrieval_query":
                        retrieval_query,
                    "query_rewritten":
                        query_was_rewritten,
                }

            # =================================================
            # 10. PARTIAL + REJECTED WEB -> DOCUMENT ONLY
            # =================================================

            if (
                evidence_status
                == EvidenceStatus.PARTIAL
                and not has_web_evidence
            ):
                source_type = "document"

                answer = self._generate_answer(
                    question=question,
                    context=document_context,
                    history=prepared_history,
                    source_type=source_type,
                )

                logger.warning(
                    "Web supplementation rejected or unavailable; "
                    "using partial document evidence "
                    "document_id=%s "
                    "retrieval_query=%r "
                    "web_reason=%s",
                    document_id,
                    retrieval_query,
                    web_validation_reason,
                )

                return {
                    "answer": answer,
                    "source_type": source_type,
                    "retrieval_results":
                        retrieval_results,
                    "web_results": [],
                    "context":
                        document_context,
                    "document_context":
                        document_context,
                    "web_context": "",
                    "document_sufficient":
                        False,
                    "evidence_status":
                        evidence_status.value,
                    "sufficiency_reason":
                        (
                            f"{sufficiency_reason} "
                            "Web supplementation was unavailable "
                            "or failed relevance validation."
                        ),
                    "web_evidence_relevant":
                        False,
                    "web_validation_reason":
                        web_validation_reason,
                    "original_question":
                        question,
                    "retrieval_query":
                        retrieval_query,
                    "query_rewritten":
                        query_was_rewritten,
                }

            # =================================================
            # 11. INSUFFICIENT + RELEVANT WEB -> WEB ONLY
            # =================================================

            if (
                evidence_status
                == EvidenceStatus.INSUFFICIENT
                and has_web_evidence
            ):
                source_type = "web"

                answer = self._generate_answer(
                    question=question,
                    context=web_context,
                    history=prepared_history,
                    source_type=source_type,
                )

                logger.info(
                    "RAG invocation completed "
                    "document_id=%s "
                    "source_type=%s "
                    "web_results=%d "
                    "context_chars=%d "
                    "query_rewritten=%s",
                    document_id,
                    source_type,
                    len(web_results),
                    len(web_context),
                    query_was_rewritten,
                )

                return {
                    "answer": answer,
                    "source_type": source_type,
                    "retrieval_results":
                        retrieval_results,
                    "web_results":
                        web_results,
                    "context":
                        web_context,
                    "document_context":
                        document_context,
                    "web_context":
                        web_context,
                    "document_sufficient":
                        False,
                    "evidence_status":
                        evidence_status.value,
                    "sufficiency_reason":
                        sufficiency_reason,
                    "web_evidence_relevant":
                        True,
                    "web_validation_reason":
                        web_validation_reason,
                    "original_question":
                        question,
                    "retrieval_query":
                        retrieval_query,
                    "query_rewritten":
                        query_was_rewritten,
                }

            # =================================================
            # 12. INSUFFICIENT + REJECTED WEB
            # =================================================

            source_type = "insufficient"

            answer = (
                "The uploaded document does not contain "
                "enough relevant information to answer this "
                "question, and no sufficiently relevant web "
                "evidence was available."
            )

            logger.warning(
                "No usable document or validated web evidence "
                "document_id=%s "
                "retrieval_query=%r "
                "web_reason=%s",
                document_id,
                retrieval_query,
                web_validation_reason,
            )

            return {
                "answer": answer,
                "source_type": source_type,
                "retrieval_results":
                    retrieval_results,
                "web_results": [],
                "context": "",
                "document_context":
                    document_context,
                "web_context": "",
                "document_sufficient":
                    False,
                "evidence_status":
                    evidence_status.value,
                "sufficiency_reason":
                    sufficiency_reason,
                "web_evidence_relevant":
                    False,
                "web_validation_reason":
                    web_validation_reason,
                "original_question":
                    question,
                "retrieval_query":
                    retrieval_query,
                "query_rewritten":
                    query_was_rewritten,
            }

        except LLMServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "RAG invocation failed "
                "document_id=%s",
                document_id,
            )

            raise LLMServiceError(
                "Unable to generate RAG answer."
            ) from exc

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def close(
        self,
    ) -> None:
        """
        Close resources owned by retrieval components.
        """

        self.retriever.close()