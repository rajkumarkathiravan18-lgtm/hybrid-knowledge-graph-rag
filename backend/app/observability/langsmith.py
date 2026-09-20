import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

from langsmith import traceable

from app.core.config import get_settings


logger = logging.getLogger(__name__)

F = TypeVar(
    "F",
    bound=Callable[..., Any],
)


# =========================================================
# LANGSMITH CONFIGURATION
# =========================================================

def configure_langsmith() -> bool:
    """
    Configure LangSmith environment variables from the
    application's central settings.

    Returns:
        True when LangSmith tracing is enabled and an API
        key is available. Otherwise False.
    """

    settings = get_settings()

    tracing_enabled = bool(
        settings.langsmith_tracing
    )

    api_key = (
        settings.langsmith_api_key
        or ""
    ).strip()

    project = (
        settings.langsmith_project
        or "hybrid-kg-rag"
    ).strip()

    # -----------------------------------------------------
    # TRACING FLAG
    # -----------------------------------------------------

    os.environ[
        "LANGSMITH_TRACING"
    ] = (
        "true"
        if tracing_enabled
        else "false"
    )

    # LangChain compatibility.
    os.environ[
        "LANGCHAIN_TRACING_V2"
    ] = (
        "true"
        if tracing_enabled
        else "false"
    )

    # -----------------------------------------------------
    # PROJECT
    # -----------------------------------------------------

    os.environ[
        "LANGSMITH_PROJECT"
    ] = project

    os.environ[
        "LANGCHAIN_PROJECT"
    ] = project

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    if api_key:
        os.environ[
            "LANGSMITH_API_KEY"
        ] = api_key

        os.environ[
            "LANGCHAIN_API_KEY"
        ] = api_key

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    configured = (
        tracing_enabled
        and bool(api_key)
    )

    if configured:
        logger.info(
            "LangSmith tracing enabled project=%s",
            project,
        )

    elif tracing_enabled:
        logger.warning(
            "LangSmith tracing requested but API key "
            "is not configured."
        )

    else:
        logger.info(
            "LangSmith tracing disabled."
        )

    return configured


# =========================================================
# TRACE DECORATOR
# =========================================================

def traced(
    *,
    name: str,
    run_type: str = "chain",
) -> Callable[[F], F]:
    """
    Application-level wrapper around LangSmith's
    @traceable decorator.

    Example:

        @traced(
            name="hybrid-retrieval",
            run_type="retriever",
        )
        def retrieve(...):
            ...

    Keeping this wrapper in one module avoids coupling the
    rest of the application directly to LangSmith setup.
    """

    def decorator(
        function: F,
    ) -> F:

        wrapped = traceable(
            name=name,
            run_type=run_type,
        )(
            function
        )

        @wraps(function)
        def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return wrapped(
                *args,
                **kwargs,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# =========================================================
# SAFE TRACE METADATA
# =========================================================

def build_trace_metadata(
    *,
    document_id: str | None = None,
    component: str | None = None,
    retrieval_type: str | None = None,
) -> dict[str, Any]:
    """
    Build non-secret metadata that may be attached to
    LangSmith traces.

    API keys, passwords, Neo4j credentials and other
    secrets must never be placed in trace metadata.
    """

    metadata: dict[str, Any] = {}

    if document_id:
        metadata[
            "document_id"
        ] = document_id

    if component:
        metadata[
            "component"
        ] = component

    if retrieval_type:
        metadata[
            "retrieval_type"
        ] = retrieval_type

    return metadata


# =========================================================
# INITIAL CONFIGURATION
# =========================================================

LANGSMITH_ENABLED = configure_langsmith()