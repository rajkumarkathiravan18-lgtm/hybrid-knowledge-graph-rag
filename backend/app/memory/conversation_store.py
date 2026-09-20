import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    """
    Persistent SQLite storage for chatbot conversations.

    Stores:
    - conversations
    - user messages
    - assistant messages

    SQLite is used for the current application because it is lightweight,
    persistent, and does not require another database service.

    The class is intentionally isolated from the RAG pipeline so the storage
    implementation can later be replaced with PostgreSQL if required.
    """

    def __init__(
        self,
        database_path: str | Path = "data/memory/conversations.db",
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    # ========================================================
    # CONNECTION
    # ========================================================

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        return connection

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_type TEXT,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation_id
                ON messages(conversation_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversations_updated_at
                ON conversations(updated_at)
                """
            )

            connection.commit()

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _generate_title(
        first_message: str | None,
    ) -> str:
        if not first_message:
            return "New conversation"

        clean_message = " ".join(
            first_message.strip().split()
        )

        if len(clean_message) <= 60:
            return clean_message

        return f"{clean_message[:57]}..."

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    def create_conversation(
        self,
        document_id: str | None = None,
        first_message: str | None = None,
    ) -> dict[str, Any]:
        conversation_id = self._generate_id()
        timestamp = self._utc_now()

        title = self._generate_title(
            first_message
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    document_id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    document_id,
                    title,
                    timestamp,
                    timestamp,
                ),
            )

            connection.commit()

        return {
            "conversation_id": conversation_id,
            "document_id": document_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def conversation_exists(
        self,
        conversation_id: str,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM conversations
                WHERE conversation_id = ?
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()

        return row is not None

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            conversation = connection.execute(
                """
                SELECT
                    conversation_id,
                    document_id,
                    title,
                    created_at,
                    updated_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            if conversation is None:
                return None

            messages = connection.execute(
                """
                SELECT
                    message_id,
                    conversation_id,
                    role,
                    content,
                    source_type,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

        result = dict(conversation)

        result["messages"] = [
            dict(message)
            for message in messages
        ]

        return result

    def list_conversations(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        safe_limit = max(
            1,
            min(limit, 100),
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    conversation_id,
                    document_id,
                    title,
                    created_at,
                    updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            connection.commit()

        return cursor.rowcount > 0

    # ========================================================
    # MESSAGES
    # ========================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        if role not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError(
                "role must be user, assistant, or system"
            )

        if not content.strip():
            raise ValueError(
                "message content cannot be empty"
            )

        if not self.conversation_exists(
            conversation_id
        ):
            raise ValueError(
                f"Conversation does not exist: "
                f"{conversation_id}"
            )

        message_id = self._generate_id()
        timestamp = self._utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    source_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content.strip(),
                    source_type,
                    timestamp,
                ),
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    timestamp,
                    conversation_id,
                ),
            )

            connection.commit()

        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content.strip(),
            "source_type": source_type,
            "created_at": timestamp,
        }

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent messages in chronological order.

        Example:
        Database contains 100 messages.
        limit=20 retrieves only the latest 20 messages so the
        entire conversation is not sent to the LLM.
        """

        safe_limit = max(
            1,
            min(limit, 100),
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM (
                    SELECT
                        message_id,
                        conversation_id,
                        role,
                        content,
                        source_type,
                        created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (
                    conversation_id,
                    safe_limit,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # DOCUMENT ASSOCIATION
    # ========================================================

    def update_document(
        self,
        conversation_id: str,
        document_id: str | None,
    ) -> bool:
        timestamp = self._utc_now()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET
                    document_id = ?,
                    updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    document_id,
                    timestamp,
                    conversation_id,
                ),
            )

            connection.commit()

        return cursor.rowcount > 0