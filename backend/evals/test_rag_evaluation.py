import json
import os
from pathlib import Path

import pytest

from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from app.core.config import get_settings
from app.llm.chain import RAGChain


# =========================================================
# PATHS
# =========================================================

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"


# =========================================================
# APPLICATION SETTINGS
# =========================================================

settings = get_settings()


# =========================================================
# DEEPEVAL ENVIRONMENT
# =========================================================

def configure_deepeval_environment() -> None:
    """
    Expose the application's existing OpenAI API key to
    DeepEval's OpenAI judge model.

    DeepEval does not require a separate OpenAI key for
    these local LLM-based evaluation metrics.
    """

    api_key = (
        settings.openai_api_key
        or ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    os.environ["OPENAI_API_KEY"] = api_key


configure_deepeval_environment()


# =========================================================
# TEST DOCUMENT
# =========================================================

DOCUMENT_ID = (
    "5d16bb10-178e-416d-bf48-23101570617c"
)


# =========================================================
# DATASET
# =========================================================

def load_dataset() -> list[dict]:
    """
    Load the local Hybrid RAG evaluation dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    if not dataset:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    return dataset


DATASET = load_dataset()


# =========================================================
# METRICS
# =========================================================

def build_rag_metrics():
    """
    Metrics for general RAG quality.

    AnswerRelevancyMetric:
        Is the generated answer relevant to the question?

    FaithfulnessMetric:
        Are answer claims supported by retrieved evidence?

    ContextualRelevancyMetric:
        Is the retrieved evidence relevant to the question?
    """

    evaluation_model = settings.llm_model

    return [
        AnswerRelevancyMetric(
            threshold=0.5,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
        ),

        FaithfulnessMetric(
            threshold=0.5,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
        ),

        ContextualRelevancyMetric(
            threshold=0.5,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
        ),
    ]


def build_hallucination_metric():
    """
    Hallucination / groundedness evaluation.

    DeepEval 4.2.x uses the standard metric direction:

        higher score = better
        lower score = worse

    Therefore 0.5 represents the minimum passing score.
    """

    return HallucinationMetric(
        threshold=0.5,
        model=settings.llm_model,
        include_reason=True,
        async_mode=False,
    )


# =========================================================
# RESULT -> EVIDENCE
# =========================================================

def result_to_context(
    result: dict,
) -> str:
    """
    Convert one HybridRetriever result into a readable
    evidence string.
    """

    retrieval_type = result.get(
        "retrieval_type"
    )

    # =====================================================
    # VECTOR RESULT
    # =====================================================

    if retrieval_type == "vector":

        return str(
            result.get(
                "text",
                "",
            )
            or ""
        ).strip()

    # =====================================================
    # GRAPH RESULT
    # =====================================================

    if retrieval_type == "graph":

        graph_source = str(
            result.get(
                "graph_source",
                "",
            )
            or ""
        ).strip()

        graph_relationship = str(
            result.get(
                "graph_relationship",
                "",
            )
            or ""
        ).strip()

        graph_target = str(
            result.get(
                "graph_target",
                "",
            )
            or ""
        ).strip()

        graph_evidence = str(
            result.get(
                "graph_evidence",
                "",
            )
            or result.get(
                "text",
                "",
            )
            or ""
        ).strip()

        parts: list[str] = []

        if (
            graph_source
            and graph_relationship
            and graph_target
        ):
            parts.append(
                f"{graph_source} "
                f"{graph_relationship} "
                f"{graph_target}"
            )

        if graph_evidence:
            parts.append(
                graph_evidence
            )

        return "\n".join(
            parts
        ).strip()

    # =====================================================
    # FALLBACK
    # =====================================================

    return str(
        result.get(
            "text",
            "",
        )
        or ""
    ).strip()


# =========================================================
# COMPLETE RETRIEVAL CONTEXT
# =========================================================

def build_retrieval_context(
    retrieval_results: list[dict],
) -> list[str]:
    """
    Build the complete retrieval context.

    This intentionally contains evidence from both:

        FAISS
        Neo4j

    It is used for evaluating retrieval and general
    faithfulness.
    """

    contexts: list[str] = []
    seen: set[str] = set()

    for result in retrieval_results:

        context_text = result_to_context(
            result
        )

        if not context_text:
            continue

        if context_text in seen:
            continue

        seen.add(
            context_text
        )

        contexts.append(
            context_text
        )

    return contexts


# =========================================================
# QUESTION-RELEVANT GROUNDING CONTEXT
# =========================================================

def build_grounding_context(
    question: str,
    retrieval_results: list[dict],
) -> list[str]:
    """
    Build focused evidence for HallucinationMetric.

    Hallucination evaluation should determine whether the
    claims made in the answer are supported by evidence.

    It should not require the answer to reproduce unrelated
    information that happened to be retrieved.

    The selection is deterministic and does not require an
    additional LLM call.
    """

    question_lower = question.lower()

    selected: list[str] = []
    seen: set[str] = set()

    # =====================================================
    # ENTITY / TECHNOLOGY FOCUS
    # =====================================================

    focus_terms: list[str] = []

    if "faiss" in question_lower:
        focus_terms.append(
            "faiss"
        )

    if "neo4j" in question_lower:
        focus_terms.append(
            "neo4j"
        )

    # If both technologies are explicitly requested,
    # preserve evidence for both.
    if not focus_terms:
        focus_terms.extend(
            [
                "faiss",
                "neo4j",
            ]
        )

    # =====================================================
    # SELECT RELEVANT EVIDENCE
    # =====================================================

    for result in retrieval_results:

        context_text = result_to_context(
            result
        )

        if not context_text:
            continue

        context_lower = (
            context_text.lower()
        )

        if any(
            term in context_lower
            for term in focus_terms
        ):
            if context_text not in seen:

                seen.add(
                    context_text
                )

                selected.append(
                    context_text
                )

    # =====================================================
    # SAFE FALLBACK
    # =====================================================

    if not selected:

        return build_retrieval_context(
            retrieval_results
        )

    return selected


# =========================================================
# RAG FIXTURE
# =========================================================

@pytest.fixture(
    scope="module"
)
def rag_chain():
    """
    Create one real RAGChain for the evaluation session.
    """

    chain = RAGChain()

    try:
        yield chain

    finally:
        chain.close()


# =========================================================
# PARAMETRIZED EVALUATION
# =========================================================

@pytest.mark.parametrize(
    "evaluation_case",
    DATASET,
    ids=[
        item["id"]
        for item in DATASET
    ],
)
def test_hybrid_rag_quality(
    evaluation_case: dict,
    rag_chain: RAGChain,
):
    """
    End-to-end Hybrid Knowledge Graph RAG evaluation.

    Evaluates:

    - Answer relevancy
    - Faithfulness
    - Contextual relevancy
    - Hallucination / groundedness
    """

    question = str(
        evaluation_case[
            "question"
        ]
    ).strip()

    expected_answer = str(
        evaluation_case[
            "expected_answer"
        ]
    ).strip()

    # =====================================================
    # REAL RAG EXECUTION
    # =====================================================

    result = rag_chain.invoke(
        document_id=DOCUMENT_ID,
        question=question,
    )

    actual_answer = str(
        result.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    retrieval_results = result.get(
        "retrieval_results",
        [],
    )

    # =====================================================
    # PIPELINE VALIDATION
    # =====================================================

    assert actual_answer, (
        "RAG pipeline returned an empty answer."
    )

    assert retrieval_results, (
        "RAG pipeline returned no retrieval evidence."
    )

    # =====================================================
    # COMPLETE RETRIEVAL CONTEXT
    # =====================================================

    retrieval_context = (
        build_retrieval_context(
            retrieval_results
        )
    )

    assert retrieval_context, (
        "Unable to build retrieval context from "
        "Hybrid RAG results."
    )

    # =====================================================
    # QUESTION-RELEVANT GROUNDING CONTEXT
    # =====================================================

    grounding_context = (
        build_grounding_context(
            question=question,
            retrieval_results=retrieval_results,
        )
    )

    assert grounding_context, (
        "Unable to build hallucination grounding context."
    )

    # =====================================================
    # GENERAL RAG QUALITY
    # =====================================================

    rag_test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context,
        metadata={
            "evaluation_id": evaluation_case["id"],
            "document_id": DOCUMENT_ID,
            "evaluation_type": "rag_quality",
        },
    )

    assert_test(
        rag_test_case,
        build_rag_metrics(),
        run_async=False,
    )

    # =====================================================
    # HALLUCINATION / GROUNDEDNESS
    # =====================================================

    hallucination_test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        context=grounding_context,
        metadata={
            "evaluation_id": evaluation_case["id"],
            "document_id": DOCUMENT_ID,
            "evaluation_type": "groundedness",
        },
    )

    assert_test(
        hallucination_test_case,
        [
            build_hallucination_metric()
        ],
        run_async=False,
    )