from typing import Any

import streamlit as st

from api_client import (
    APIClient,
    APIClientError,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid Knowledge Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        min-width: 330px;
        max-width: 330px;
    }

    .app-subtitle {
        color: #777;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    .source-type {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        border: 1px solid rgba(128, 128, 128, 0.30);
        font-size: 0.78rem;
        margin-bottom: 8px;
    }

    .conversation-title {
        font-size: 0.85rem;
        opacity: 0.8;
    }

    .small-muted {
        opacity: 0.7;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# API CLIENT
# =========================================================


@st.cache_resource
def get_api_client() -> APIClient:
    return APIClient()


api = get_api_client()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "backend_online" not in st.session_state:
    st.session_state.backend_online = False


# =========================================================
# DOCUMENT HELPERS
# =========================================================


def refresh_documents() -> None:
    try:
        st.session_state.documents = (
            api.list_documents()
        )

    except APIClientError as exc:
        st.session_state.documents = []

        st.error(
            f"Unable to load documents: {exc.message}"
        )


def get_document_id(
    document: dict[str, Any],
) -> str:
    return str(
        document.get(
            "document_id",
            "",
        )
    )


def get_document_filename(
    document: dict[str, Any],
) -> str:
    return str(
        document.get(
            "filename",
            "Unnamed document",
        )
    )


def find_selected_document() -> dict | None:
    selected_id = (
        st.session_state.selected_document_id
    )

    if not selected_id:
        return None

    for document in st.session_state.documents:
        if (
            get_document_id(document)
            == selected_id
        ):
            return document

    return None


# =========================================================
# CONVERSATION HELPERS
# =========================================================


def new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


def refresh_conversations() -> None:
    if not st.session_state.backend_online:
        st.session_state.conversations = []
        return

    try:
        st.session_state.conversations = (
            api.list_conversations()
        )

    except APIClientError as exc:
        st.session_state.conversations = []

        st.error(
            "Unable to load chat history: "
            f"{exc.message}"
        )


def select_document(
    document_id: str | None,
) -> None:
    previous = (
        st.session_state.selected_document_id
    )

    if previous != document_id:
        st.session_state.selected_document_id = (
            document_id
        )

        new_chat()


def load_conversation(
    conversation_id: str,
) -> None:
    """
    Load a persisted conversation from FastAPI and
    reconstruct Streamlit chat history.

    Persisted messages contain answer text and source_type.
    Detailed retrieval provenance belongs to individual
    live chat responses and is therefore not fabricated
    when restoring historical messages.
    """

    conversation = api.get_conversation(
        conversation_id
    )

    conversation_document_id = (
        conversation.get(
            "document_id"
        )
    )

    if conversation_document_id:
        st.session_state.selected_document_id = (
            conversation_document_id
        )

    st.session_state.conversation_id = (
        conversation_id
    )

    restored_messages = []

    for message in conversation.get(
        "messages",
        [],
    ):
        role = str(
            message.get(
                "role",
                "assistant",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        if not content:
            continue

        restored_messages.append(
            {
                "role": role,
                "content": content,
                "source_type": (
                    message.get(
                        "source_type"
                    )
                ),
                "sources": [],
                "graph_context": [],
                "web_sources": [],
            }
        )

    st.session_state.messages = (
        restored_messages
    )


# =========================================================
# PROVENANCE RENDERING
# =========================================================


def render_source_type(
    source_type: str | None,
) -> None:
    if not source_type:
        return

    labels = {
        "document":
            "Document grounded",

        "web":
            "Web grounded",

        "document+web":
            "Document + Web grounded",

        "insufficient":
            "Insufficient evidence",
    }

    label = labels.get(
        source_type,
        source_type,
    )

    st.markdown(
        (
            '<div class="source-type">'
            f"{label}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_sources(
    sources: list[dict],
) -> None:
    if not sources:
        return

    with st.expander(
        f"Document Sources ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            filename = source.get(
                "filename"
            ) or "Unknown document"

            chunk_id = source.get(
                "chunk_id"
            ) or "Unknown chunk"

            page_number = source.get(
                "page_number"
            )

            score = source.get(
                "score"
            )

            st.markdown(
                f"**{index}. {filename}**"
            )

            st.caption(
                f"Chunk: {chunk_id}"
            )

            if page_number is not None:
                st.caption(
                    f"Page: {page_number}"
                )

            if score is not None:
                try:
                    st.caption(
                        "Retrieval score: "
                        f"{float(score):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if index < len(sources):
                st.divider()


def render_graph_evidence(
    graph_context: list[dict],
) -> None:
    if not graph_context:
        return

    with st.expander(
        (
            "Knowledge Graph Evidence "
            f"({len(graph_context)})"
        ),
        expanded=False,
    ):
        for index, evidence in enumerate(
            graph_context,
            start=1,
        ):
            source = evidence.get(
                "source",
                "Unknown",
            )

            relationship = evidence.get(
                "relationship",
                "RELATED_TO",
            )

            target = evidence.get(
                "target",
                "Unknown",
            )

            supporting_text = (
                evidence.get(
                    "evidence"
                )
            )

            st.markdown(
                (
                    f"**{source}** "
                    f"→ `{relationship}` → "
                    f"**{target}**"
                )
            )

            if supporting_text:
                st.caption(
                    supporting_text
                )

            chunk_id = evidence.get(
                "chunk_id"
            )

            if chunk_id:
                st.caption(
                    f"Chunk: {chunk_id}"
                )

            score = evidence.get(
                "score"
            )

            if score is not None:
                try:
                    st.caption(
                        "Graph score: "
                        f"{float(score):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if index < len(graph_context):
                st.divider()


def render_web_sources(
    web_sources: list[dict],
) -> None:
    if not web_sources:
        return

    with st.expander(
        f"Web Sources ({len(web_sources)})",
        expanded=False,
    ):
        for index, source in enumerate(
            web_sources,
            start=1,
        ):
            title = (
                source.get("title")
                or "Web source"
            )

            url = str(
                source.get(
                    "url",
                    "",
                )
            ).strip()

            snippet = (
                source.get(
                    "snippet"
                )
            )

            if url:
                st.markdown(
                    f"**{index}. [{title}]({url})**"
                )

            else:
                st.markdown(
                    f"**{index}. {title}**"
                )

            if snippet:
                st.caption(
                    snippet
                )

            if index < len(web_sources):
                st.divider()


def render_assistant_provenance(
    message: dict,
) -> None:
    render_source_type(
        message.get(
            "source_type"
        )
    )

    render_sources(
        message.get(
            "sources",
            [],
        )
    )

    render_graph_evidence(
        message.get(
            "graph_context",
            [],
        )
    )

    render_web_sources(
        message.get(
            "web_sources",
            [],
        )
    )


# =========================================================
# BACKEND HEALTH
# =========================================================

try:
    health = api.health()

    st.session_state.backend_online = (
        health.get("status")
        == "healthy"
    )

except APIClientError:
    health = {}
    st.session_state.backend_online = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🧠 Hybrid KG RAG"
    )

    st.caption(
        "Vector + Knowledge Graph + Web Retrieval"
    )

    st.divider()

    # -----------------------------------------------------
    # SYSTEM
    # -----------------------------------------------------

    st.subheader(
        "System"
    )

    if st.session_state.backend_online:
        st.success(
            "Backend connected"
        )

    else:
        st.error(
            "Backend unavailable"
        )

        st.caption(
            (
                "Start FastAPI on port 8000 "
                "and refresh this page."
            )
        )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENT UPLOAD
    # -----------------------------------------------------

    st.subheader(
        "Upload document"
    )

    uploaded_file = st.file_uploader(
        "PDF, TXT or Markdown",
        type=[
            "pdf",
            "txt",
            "md",
        ],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:

        if st.button(
            "Process document",
            type="primary",
            use_container_width=True,
        ):

            if not st.session_state.backend_online:
                st.error(
                    "Backend is unavailable."
                )

            else:
                try:
                    with st.spinner(
                        (
                            "Processing document, "
                            "creating embeddings and "
                            "building the knowledge graph..."
                        )
                    ):
                        result = (
                            api.upload_document(
                                filename=(
                                    uploaded_file.name
                                ),
                                file_bytes=(
                                    uploaded_file.getvalue()
                                ),
                                content_type=(
                                    uploaded_file.type
                                ),
                            )
                        )

                    new_document_id = (
                        result.get(
                            "document_id"
                        )
                    )

                    if new_document_id:
                        st.session_state.selected_document_id = (
                            new_document_id
                        )

                    new_chat()
                    refresh_documents()
                    refresh_conversations()

                    st.success(
                        "Document processed successfully."
                    )

                    st.rerun()

                except APIClientError as exc:
                    st.error(
                        exc.message
                    )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENTS
    # -----------------------------------------------------

    st.subheader(
        "Documents"
    )

    if st.session_state.backend_online:
        refresh_documents()

    documents = (
        st.session_state.documents
    )

    if documents:

        document_map = {
            get_document_id(document):
                get_document_filename(document)

            for document in documents
        }

        document_ids = list(
            document_map.keys()
        )

        current_id = (
            st.session_state.selected_document_id
        )

        if current_id not in document_ids:
            current_id = document_ids[0]

            st.session_state.selected_document_id = (
                current_id
            )

            new_chat()

        selected_index = (
            document_ids.index(
                current_id
            )
        )

        selected_id = st.selectbox(
            "Active document",
            options=document_ids,
            index=selected_index,
            format_func=lambda document_id: (
                document_map.get(
                    document_id,
                    document_id,
                )
            ),
        )

        select_document(
            selected_id
        )

        selected_document = (
            find_selected_document()
        )

        if selected_document:

            document_status = str(
                selected_document.get(
                    "status",
                    "unknown",
                )
            ).lower()

            chunk_count = (
                selected_document.get(
                    "chunk_count",
                    0,
                )
            )

            if document_status == "ready":
                st.success(
                    (
                        "Ready · "
                        f"{chunk_count} chunk(s)"
                    )
                )

            elif document_status == "processing":
                st.warning(
                    "Processing..."
                )

            elif document_status == "failed":
                st.error(
                    "Processing failed"
                )

                error_message = (
                    selected_document.get(
                        "error_message"
                    )
                )

                if error_message:
                    st.caption(
                        error_message
                    )

            else:
                st.info(
                    f"Status: {document_status}"
                )

        if st.button(
            "＋ New chat",
            use_container_width=True,
        ):
            new_chat()
            st.rerun()

        # -------------------------------------------------
        # CHAT HISTORY
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Chat history"
        )

        refresh_conversations()

        selected_document_conversations = [
            conversation
            for conversation
            in st.session_state.conversations
            if conversation.get(
                "document_id"
            ) == selected_id
        ]

        if selected_document_conversations:

            for conversation in (
                selected_document_conversations
            ):
                conversation_id = str(
                    conversation.get(
                        "conversation_id",
                        "",
                    )
                )

                title = str(
                    conversation.get(
                        "title",
                        "Untitled chat",
                    )
                )

                is_active = (
                    conversation_id
                    == st.session_state.conversation_id
                )

                button_label = (
                    f"● {title}"
                    if is_active
                    else title
                )

                if st.button(
                    button_label,
                    key=(
                        "conversation_"
                        f"{conversation_id}"
                    ),
                    use_container_width=True,
                ):
                    try:
                        load_conversation(
                            conversation_id
                        )

                        st.rerun()

                    except APIClientError as exc:
                        st.error(
                            exc.message
                        )

        else:
            st.caption(
                "No saved chats for this document."
            )

        # -------------------------------------------------
        # ACTIVE CHAT OPTIONS
        # -------------------------------------------------

        if st.session_state.conversation_id:

            with st.expander(
                "Current chat options"
            ):
                st.caption(
                    (
                        "Deleting this chat removes "
                        "its persistent conversation "
                        "history."
                    )
                )

                if st.button(
                    "Delete current chat",
                    use_container_width=True,
                ):
                    try:
                        api.delete_conversation(
                            st.session_state.conversation_id
                        )

                        new_chat()
                        refresh_conversations()

                        st.success(
                            "Conversation deleted."
                        )

                        st.rerun()

                    except APIClientError as exc:
                        st.error(
                            exc.message
                        )

        # -------------------------------------------------
        # DOCUMENT OPTIONS
        # -------------------------------------------------

        with st.expander(
            "Document options"
        ):

            st.caption(
                (
                    "Deleting removes the uploaded "
                    "document, FAISS index and its "
                    "Neo4j evidence."
                )
            )

            confirm_delete = st.checkbox(
                "I want to delete this document"
            )

            if st.button(
                "Delete document",
                disabled=not confirm_delete,
                use_container_width=True,
            ):

                try:
                    with st.spinner(
                        "Deleting document..."
                    ):
                        api.delete_document(
                            selected_id
                        )

                    st.session_state.selected_document_id = (
                        None
                    )

                    new_chat()
                    refresh_documents()
                    refresh_conversations()

                    st.success(
                        "Document deleted."
                    )

                    st.rerun()

                except APIClientError as exc:
                    st.error(
                        exc.message
                    )

    else:
        st.info(
            "No documents uploaded yet."
        )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "Hybrid Knowledge Graph RAG"
)

st.markdown(
    """
    <div class="app-subtitle">
        Ask grounded questions using document vectors,
        Knowledge Graph evidence and controlled web fallback.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NO BACKEND
# =========================================================

if not st.session_state.backend_online:

    st.error(
        (
            "The FastAPI backend is not available. "
            "Start the backend before using the chatbot."
        )
    )

    st.stop()


# =========================================================
# NO DOCUMENT
# =========================================================

selected_document = (
    find_selected_document()
)

if selected_document is None:

    st.info(
        (
            "Upload a document from the sidebar "
            "to start chatting."
        )
    )

    st.stop()


# =========================================================
# DOCUMENT HEADER
# =========================================================

filename = get_document_filename(
    selected_document
)

status = str(
    selected_document.get(
        "status",
        "unknown",
    )
).lower()

chunk_count = selected_document.get(
    "chunk_count",
    0,
)

header_col_1, header_col_2 = (
    st.columns(
        [4, 1]
    )
)

with header_col_1:
    st.markdown(
        f"### 📄 {filename}"
    )

with header_col_2:

    if status == "ready":
        st.success(
            "Ready"
        )

    elif status == "processing":
        st.warning(
            "Processing"
        )

    elif status == "failed":
        st.error(
            "Failed"
        )

    else:
        st.info(
            status.title()
        )

st.caption(
    (
        f"{chunk_count} chunk(s) · "
        f"Document ID: "
        f"{selected_document.get('document_id')}"
    )
)

if st.session_state.conversation_id:
    st.caption(
        (
            "Conversation ID: "
            f"{st.session_state.conversation_id}"
        )
    )

st.divider()


# =========================================================
# DOCUMENT NOT READY
# =========================================================

if status != "ready":

    if status == "processing":
        st.warning(
            (
                "This document is still processing. "
                "Wait until its status becomes READY."
            )
        )

    elif status == "failed":
        st.error(
            (
                "This document failed during processing. "
                "Check the processing error before chatting."
            )
        )

    else:
        st.info(
            (
                "This document is not ready "
                "for chat yet."
            )
        )

    st.stop()


# =========================================================
# EMPTY CHAT
# =========================================================

if not st.session_state.messages:

    st.info(
        (
            "The document is ready. "
            "Start a new conversation below."
        )
    )

    st.markdown(
        "**Example questions**"
    )

    st.markdown(
        """
        - What are the main concepts in this document?
        - Explain the relationships between the key entities.
        - Summarize the most relevant information.
        - Compare the document with current information from the web.
        """
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(
        role
    ):
        st.markdown(
            content
        )

        if role == "assistant":
            render_assistant_provenance(
                message
            )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question..."
)


# =========================================================
# PROCESS CHAT
# =========================================================

if question:

    document_id = str(
        selected_document.get(
            "document_id",
            "",
        )
    )

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):

        try:
            with st.spinner(
                (
                    "Searching document, graph "
                    "and web evidence..."
                )
            ):

                response = api.chat(
                    document_id=document_id,
                    question=question,
                    conversation_id=(
                        st.session_state.conversation_id
                    ),
                )

            answer = response.get(
                "answer",
                (
                    "The backend returned "
                    "an empty answer."
                ),
            )

            conversation_id = (
                response.get(
                    "conversation_id"
                )
            )

            source_type = response.get(
                "source_type",
                "insufficient",
            )

            sources = response.get(
                "sources",
                [],
            )

            graph_context = response.get(
                "graph_context",
                [],
            )

            web_sources = response.get(
                "web_sources",
                [],
            )

            if conversation_id:
                st.session_state.conversation_id = (
                    conversation_id
                )

            st.markdown(
                answer
            )

            render_source_type(
                source_type
            )

            render_sources(
                sources
            )

            render_graph_evidence(
                graph_context
            )

            render_web_sources(
                web_sources
            )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "source_type": source_type,
                "sources": sources,
                "graph_context":
                    graph_context,
                "web_sources":
                    web_sources,
            }

            st.session_state.messages.append(
                assistant_message
            )

            refresh_conversations()

        except APIClientError as exc:

            error_message = (
                f"Request failed: {exc.message}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "source_type": None,
                    "sources": [],
                    "graph_context": [],
                    "web_sources": [],
                }
            )