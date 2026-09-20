from fastapi import APIRouter, HTTPException, Query, status

from app.core.logging import get_logger
from app.models.api import (
    ConversationDetail,
    ConversationSummary,
)
from app.services.chat_service import ChatService


logger = get_logger(__name__)


router = APIRouter(
    prefix="/api/conversations",
    tags=["Conversations"],
)


# ============================================================
# LIST CONVERSATIONS
# ============================================================


@router.get(
    "",
    response_model=list[ConversationSummary],
)
def list_conversations(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
) -> list[ConversationSummary]:
    """
    Return recent persisted conversations.

    Conversations are ordered by the persistence layer,
    normally with the most recently updated conversation
    first.
    """

    service = ChatService()

    try:
        conversations = (
            service.list_conversations(
                limit=limit
            )
        )

        return [
            ConversationSummary.model_validate(
                conversation
            )
            for conversation in conversations
        ]

    except Exception:
        logger.exception(
            "Unable to list conversations."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve conversations."
            ),
        )

    finally:
        service.close()


# ============================================================
# GET CONVERSATION
# ============================================================


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
)
def get_conversation(
    conversation_id: str,
) -> ConversationDetail:
    """
    Return one persisted conversation including its
    complete stored message history.
    """

    conversation_id = (
        conversation_id.strip()
    )

    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation ID cannot be empty."
            ),
        )

    service = ChatService()

    try:
        conversation = (
            service.get_conversation(
                conversation_id
            )
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found."
                ),
            )

        return (
            ConversationDetail.model_validate(
                conversation
            )
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Unable to retrieve conversation "
            "conversation_id=%s",
            conversation_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve conversation."
            ),
        )

    finally:
        service.close()


# ============================================================
# DELETE CONVERSATION
# ============================================================


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    conversation_id: str,
) -> dict[str, str]:
    """
    Delete one persisted conversation and all messages
    belonging to it.
    """

    conversation_id = (
        conversation_id.strip()
    )

    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation ID cannot be empty."
            ),
        )

    service = ChatService()

    try:
        deleted = (
            service.delete_conversation(
                conversation_id
            )
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found."
                ),
            )

        logger.info(
            "Conversation deleted "
            "conversation_id=%s",
            conversation_id,
        )

        return {
            "message": (
                "Conversation deleted successfully."
            )
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Unable to delete conversation "
            "conversation_id=%s",
            conversation_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to delete conversation."
            ),
        )

    finally:
        service.close()