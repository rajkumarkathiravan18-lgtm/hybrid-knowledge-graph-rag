from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_conversations import (
    router as conversations_router,
)
from app.api.routes_documents import (
    router as documents_router,
)
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


# =========================================================
# APPLICATION LIFESPAN
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle.
    """

    logger.info(
        "Starting %s environment=%s",
        settings.app_name,
        settings.app_env,
    )

    settings.ensure_directories()

    yield

    logger.info(
        "Shutting down %s",
        settings.app_name,
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================


app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the Hybrid Knowledge "
        "Graph RAG application."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# EXCEPTION HANDLERS
# =========================================================


@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """
    Handle expected application-level errors.

    Internal exception details are logged server-side,
    while the API receives a controlled response.
    """

    logger.warning(
        "Application error: method=%s path=%s "
        "status=%s code=%s message=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.error_code,
        exc.message,
    )

    response_content = {
        "error": exc.error_code,
        "message": exc.message,
    }

    if exc.details:
        response_content["details"] = (
            exc.details
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=response_content,
    )


@app.exception_handler(
    RequestValidationError
)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle invalid FastAPI/Pydantic request data.
    """

    logger.warning(
        "Request validation failed: "
        "method=%s path=%s",
        request.method,
        request.url.path,
    )

    errors = []

    for error in exc.errors():
        errors.append(
            {
                "location": [
                    str(value)
                    for value
                    in error.get(
                        "loc",
                        [],
                    )
                ],
                "message": error.get(
                    "msg",
                    "Invalid request value.",
                ),
                "type": error.get(
                    "type",
                    "validation_error",
                ),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": (
                "The request contains "
                "invalid data."
            ),
            "details": errors,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Final safety net for unexpected exceptions.

    Internal implementation details are never exposed
    to the API client.
    """

    logger.exception(
        "Unhandled exception: "
        "method=%s path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": (
                "An unexpected internal "
                "server error occurred."
            ),
        },
    )


# =========================================================
# ROUTERS
# =========================================================


app.include_router(
    health_router
)

app.include_router(
    documents_router
)

app.include_router(
    chat_router
)

app.include_router(
    conversations_router
)


# =========================================================
# ROOT
# =========================================================


@app.get(
    "/",
    tags=["Root"],
)
async def root() -> dict[str, str]:
    """
    Basic API root endpoint.
    """

    return {
        "message": (
            "Hybrid Knowledge Graph RAG API"
        )
    }