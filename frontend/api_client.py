import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)


# =========================================================
# API CLIENT ERROR
# =========================================================


class APIClientError(Exception):
    """
    Safe frontend exception for errors returned by
    the FastAPI backend or caused by HTTP communication.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.status_code = status_code
        self.details = details


# =========================================================
# API CLIENT
# =========================================================


class APIClient:
    """
    HTTP client used by Streamlit to communicate
    with the FastAPI backend.

    Streamlit communicates with backend services
    only through this client.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        configured_url = (
            base_url
            or os.getenv(
                "BACKEND_URL",
                "http://127.0.0.1:8000",
            )
        )

        self.base_url = (
            configured_url.rstrip("/")
        )

        self.timeout = timeout

    # =====================================================
    # INTERNAL REQUEST
    # =====================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict | list:
        """
        Send an HTTP request to FastAPI and normalize
        backend/network errors for Streamlit.
        """

        url = (
            f"{self.base_url}"
            f"{endpoint}"
        )

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )

        except requests.Timeout as exc:
            raise APIClientError(
                "The backend request timed out."
            ) from exc

        except requests.ConnectionError as exc:
            raise APIClientError(
                (
                    "Unable to connect to the backend. "
                    "Make sure FastAPI is running."
                )
            ) from exc

        except requests.RequestException as exc:
            raise APIClientError(
                "Unable to communicate with the backend."
            ) from exc

        # -------------------------------------------------
        # PARSE RESPONSE
        # -------------------------------------------------

        try:
            payload = response.json()

        except ValueError:
            payload = None

        # -------------------------------------------------
        # ERROR RESPONSE
        # -------------------------------------------------

        if not response.ok:

            if isinstance(
                payload,
                dict,
            ):
                message = payload.get(
                    "message"
                )

                details = payload.get(
                    "details"
                )

                if not message:
                    message = payload.get(
                        "detail"
                    )

            else:
                message = None
                details = None

            raise APIClientError(
                message=(
                    message
                    or (
                        "Backend request failed "
                        f"with status "
                        f"{response.status_code}."
                    )
                ),
                status_code=response.status_code,
                details=details,
            )

        # -------------------------------------------------
        # SUCCESS RESPONSE
        # -------------------------------------------------

        if payload is None:
            return {}

        return payload

    # =====================================================
    # HEALTH
    # =====================================================

    def health(
        self,
    ) -> dict:
        """
        Check FastAPI backend health.
        """

        result = self._request(
            "GET",
            "/health",
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid health response from backend."
            )

        return result

    # =====================================================
    # LIST DOCUMENTS
    # =====================================================

    def list_documents(
        self,
    ) -> list[dict]:
        """
        Retrieve documents known to the backend.
        """

        result = self._request(
            "GET",
            "/api/documents",
        )

        if isinstance(
            result,
            list,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            documents = result.get(
                "documents",
                [],
            )

            if isinstance(
                documents,
                list,
            ):
                return documents

        raise APIClientError(
            "Invalid document list response."
        )

    # =====================================================
    # DOCUMENT STATUS
    # =====================================================

    def get_document_status(
        self,
        document_id: str,
    ) -> dict:
        """
        Retrieve processing status for one document.
        """

        document_id = (
            document_id.strip()
        )

        if not document_id:
            raise APIClientError(
                "Document ID cannot be empty."
            )

        result = self._request(
            "GET",
            (
                "/api/documents/"
                f"{document_id}/status"
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid document status response."
            )

        return result

    # =====================================================
    # UPLOAD DOCUMENT
    # =====================================================

    def upload_document(
        self,
        filename: str,
        file_bytes: bytes,
        content_type: str | None = None,
    ) -> dict:
        """
        Upload a document to FastAPI.

        FastAPI is responsible for:
        - validation
        - parsing
        - chunking
        - FAISS indexing
        - graph extraction
        - Neo4j persistence
        """

        if not filename.strip():
            raise APIClientError(
                "Filename cannot be empty."
            )

        if not file_bytes:
            raise APIClientError(
                "Uploaded file is empty."
            )

        files = {
            "file": (
                filename,
                file_bytes,
                (
                    content_type
                    or "application/octet-stream"
                ),
            )
        }

        result = self._request(
            "POST",
            "/api/documents/upload",
            files=files,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid upload response."
            )

        return result

    # =====================================================
    # CHAT
    # =====================================================

    def chat(
        self,
        document_id: str,
        question: str,
        conversation_id: str | None = None,
    ) -> dict:
        """
        Send a user question to the Hybrid RAG backend.

        If conversation_id is supplied, FastAPI continues
        the existing persistent conversation.

        If conversation_id is omitted, FastAPI creates
        a new conversation.
        """

        document_id = (
            document_id.strip()
        )

        question = (
            question.strip()
        )

        if not document_id:
            raise APIClientError(
                "Select a document before chatting."
            )

        if not question:
            raise APIClientError(
                "Question cannot be empty."
            )

        payload = {
            "document_id":
                document_id,

            "question":
                question,
        }

        if conversation_id:
            cleaned_conversation_id = (
                conversation_id.strip()
            )

            if cleaned_conversation_id:
                payload["conversation_id"] = (
                    cleaned_conversation_id
                )

        result = self._request(
            "POST",
            "/api/chat",
            json=payload,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid chat response."
            )

        return result

    # =====================================================
    # LIST CONVERSATIONS
    # =====================================================

    def list_conversations(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """
        Retrieve recent persistent conversations.
        """

        if limit < 1:
            raise APIClientError(
                "Conversation limit must be positive."
            )

        result = self._request(
            "GET",
            "/api/conversations",
            params={
                "limit": min(
                    limit,
                    100,
                )
            },
        )

        if isinstance(
            result,
            list,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            conversations = result.get(
                "conversations"
            )

            if isinstance(
                conversations,
                list,
            ):
                return conversations

        raise APIClientError(
            "Invalid conversation list response."
        )

    # =====================================================
    # GET CONVERSATION
    # =====================================================

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict:
        """
        Retrieve one persistent conversation including
        its stored message history.
        """

        conversation_id = (
            conversation_id.strip()
        )

        if not conversation_id:
            raise APIClientError(
                "Conversation ID cannot be empty."
            )

        result = self._request(
            "GET",
            (
                "/api/conversations/"
                f"{conversation_id}"
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid conversation response."
            )

        return result

    # =====================================================
    # DELETE CONVERSATION
    # =====================================================

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> dict:
        """
        Delete one persistent conversation and its
        stored messages.
        """

        conversation_id = (
            conversation_id.strip()
        )

        if not conversation_id:
            raise APIClientError(
                "Conversation ID cannot be empty."
            )

        result = self._request(
            "DELETE",
            (
                "/api/conversations/"
                f"{conversation_id}"
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid conversation delete response."
            )

        return result

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    def delete_document(
        self,
        document_id: str,
    ) -> dict:
        """
        Delete a document and its associated
        FAISS/Neo4j resources.
        """

        document_id = (
            document_id.strip()
        )

        if not document_id:
            raise APIClientError(
                "Document ID cannot be empty."
            )

        result = self._request(
            "DELETE",
            (
                "/api/documents/"
                f"{document_id}"
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise APIClientError(
                "Invalid delete response."
            )

        return result