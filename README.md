# Hybrid Knowledge Graph RAG

A production-oriented **Hybrid Knowledge Graph Retrieval-Augmented Generation (RAG)** application that combines semantic vector retrieval using **FAISS** with graph-based retrieval using **Neo4j**.

The system supports document ingestion, entity and relationship extraction, hybrid retrieval, conversational memory, evidence sufficiency checking, controlled web fallback, LLM generation, guardrails, evaluation, observability, and production deployment using Docker.

## Live Application

**Production:** https://www.hybridrag.space

---

# 1. Project Overview

Traditional vector-only RAG systems are effective at semantic similarity search, but they can struggle with:

- relationships between entities
- multi-hop information
- structured knowledge
- connected concepts
- conversational follow-up questions
- insufficient document evidence
- hallucination control

This project extends standard RAG by combining:

1. **FAISS Vector Retrieval**
2. **Neo4j Knowledge Graph Retrieval**
3. **Hybrid Retrieval Fusion**
4. **Conversation Memory**
5. **Query Rewriting**
6. **Evidence Sufficiency Classification**
7. **Controlled Web Search Fallback**
8. **Grounded LLM Generation**
9. **Input and Output Guardrails**
10. **Evaluation and Observability**

The application is exposed through a Streamlit interface backed by a FastAPI service.

---

# 2. High-Level Architecture

```text
USER
 ↓
HTTPS / Traefik
 ↓
Streamlit Frontend
 ↓
FastAPI Backend
 ↓
Input Guardrail
 ↓
Conversation Memory
 ↓
Query Rewriting
 ↓
Hybrid Retrieval
 ├── FAISS
 └── Neo4j
 ↓
Retrieval Fusion
 ↓
Evidence Sufficiency
 ├── SUFFICIENT → Document Context
 ├── PARTIAL → Document + Web
 └── INSUFFICIENT → Web Fallback
 ↓
Context Builder
 ↓
LangChain Messages
 ↓
LLM
 ↓
Output Guardrail
 ↓
Answer + Source Evidence
```

---

# 3. Document Ingestion

The ingestion pipeline performs:

```text
Upload
 ↓
Validation
 ↓
Parsing
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Metadata Generation
 ↓
Entity Extraction
 ↓
Relationship Extraction
 ↓
FAISS Storage + Neo4j Storage
```

Metadata can include:

- document ID
- filename
- file type
- chunk ID
- page number
- source

Documents are processed once and their generated knowledge is persisted.

---

# 4. Knowledge Graph

The Knowledge Graph is stored using **Neo4j**.

Controlled entity types include:

- Person
- Organization
- Location
- Concept
- Product
- Technology
- Event
- Document

Relationship types include:

- USES
- RELATED_TO
- PART_OF
- CREATED_BY
- LOCATED_IN
- INTEGRATES_WITH
- ASSOCIATED_WITH
- MENTIONS

Neo4j is accessed using the official Neo4j Python driver and parameterized Cypher queries.

---

# 5. Vector Retrieval

The vector retrieval layer uses **FAISS**.

Document chunks are converted into embeddings and stored in a persistent FAISS index.

Default embedding model:

```text
text-embedding-3-small
```

---

# 6. Hybrid Retrieval

The system combines FAISS semantic retrieval with Neo4j graph retrieval.

Default weighting:

```text
Vector Weight = 0.60
Graph Weight  = 0.40
```

The retrieval layer performs vector retrieval, graph retrieval, result fusion, duplicate removal, ranking, metadata preservation, and context-size control.

---

# 7. Conversation Memory

The application supports persistent conversational memory using **SQLite**.

Memory stores conversations and messages, enabling contextual follow-up questions such as:

```text
User:
What architecture does this document describe?

Assistant:
...

User:
Explain that in simpler terms.
```

Conversation memory is persisted through a Docker volume.

---

# 8. Query Rewriting

Conversational follow-up questions can be rewritten into standalone retrieval queries using conversation history.

This improves retrieval accuracy for multi-turn conversations.

---

# 9. Evidence Sufficiency

Retrieved document evidence is classified into three states:

```text
SUFFICIENT
PARTIAL
INSUFFICIENT
```

**SUFFICIENT:** answer from document evidence.

**PARTIAL:** combine useful document evidence with validated web evidence.

**INSUFFICIENT:** use controlled web fallback when relevant evidence is available.

---

# 10. Controlled Web Search Fallback

Web search is controlled using:

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
```

Web results are validated for relevance before being included in the LLM context.

---

# 11. Context Management

Example configuration:

```env
MAX_CONTEXT_CHARS=12000
```

The system limits context size to help control token consumption, latency, irrelevant context, and model cost.

---

# 12. LLM Integration

The application uses LangChain messages:

```text
SystemMessage
HumanMessage
AIMessage
```

Default model:

```text
gpt-4o-mini
```

The final generation context can include the system prompt, conversation history, retrieved evidence, and user question.

---

# 13. Guardrails

The project integrates **NVIDIA NeMo Guardrails**.

```text
User Question
 ↓
Input Guardrail
 ↓
RAG Pipeline
 ↓
LLM
 ↓
Output Guardrail
 ↓
User
```

---

# 14. Evaluation

The project includes an evaluation layer using **DeepEval**.

Evaluation can measure characteristics such as:

- answer relevancy
- faithfulness
- contextual relevance
- retrieval quality

Evaluation is separated from the normal production request path.

---

# 15. Observability

The application integrates **LangSmith** for tracing and observability.

It can be used to inspect model calls, prompts, retrieval operations, chain execution, latency, failures, and token usage.

---

# 16. Backend API

The backend is built using **FastAPI**.

Important functionality includes:

- health checks
- document upload
- document processing status
- chat
- document management
- conversation management
- conversation deletion

The production FastAPI service is not directly exposed to the public internet.

---

# 17. Frontend

The user interface is built using **Streamlit**.

It provides:

- document upload
- document processing
- chat
- new conversations
- conversation history
- document evidence
- graph evidence
- web evidence
- source display

---

# 18. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM Orchestration | LangChain |
| LLM | OpenAI |
| Embeddings | OpenAI Embeddings |
| Vector Database | FAISS |
| Knowledge Graph | Neo4j |
| Graph Query | Cypher |
| Conversation Memory | SQLite |
| Web Search | DDGS |
| Guardrails | NVIDIA NeMo Guardrails |
| Evaluation | DeepEval |
| Observability | LangSmith |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Reverse Proxy | Traefik |
| TLS | Let's Encrypt |
| Version Control | Git / GitHub |
| Deployment | Ubuntu VPS |

---

# 19. Project Structure

```text
hybrid-knowledge-graph-rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── evaluation/
│   │   ├── graph/
│   │   ├── guardrails/
│   │   ├── ingestion/
│   │   ├── llm/
│   │   ├── memory/
│   │   ├── retrieval/
│   │   ├── vectorstore/
│   │   ├── web/
│   │   └── main.py
│   ├── data/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── Dockerfile
│   └── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

# 20. Environment Configuration

Create the environment file:

```bash
cp .env.example .env
```

Configure the required values. Never commit `.env`.

Key configuration areas include:

```env
OPENAI_API_KEY=
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=hybrid-kg-rag

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_URL=http://localhost:8000

MAX_UPLOAD_MB=20
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

VECTOR_TOP_K=5
GRAPH_TOP_K=5
VECTOR_WEIGHT=0.60
GRAPH_WEIGHT=0.40
MAX_CONTEXT_CHARS=12000

WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5

UPLOAD_DIR=data/uploads
FAISS_DIR=data/faiss
MEMORY_DB_PATH=data/memory/conversations.db
MEMORY_HISTORY_LIMIT=20

APP_NAME=Hybrid Knowledge Graph RAG
APP_ENV=development
DEBUG=false
```

---

# 21. Local Development

Clone the repository:

```bash
git clone https://github.com/rajkumarkathiravan18-lgtm/hybrid-knowledge-graph-rag.git
cd hybrid-knowledge-graph-rag
```

Configure the environment:

```bash
cp .env.example .env
```

Add the required credentials to `.env`.

---

# 22. Docker Deployment

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up -d
```

Check:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

Persistent Docker volumes protect application data when containers are recreated.

---

# 23. Persistent Storage

Production uses persistent Docker volumes for:

```text
Neo4j Data
Neo4j Logs
Uploaded Documents
FAISS Indexes
Conversation Memory
```

Do not use:

```bash
docker compose down -v
```

unless you intentionally want to delete the associated persistent volumes.

---

# 24. Production Deployment Architecture

```text
Internet
 ↓
hybridrag.space
 ↓
DNS
 ↓
Ubuntu VPS
 ↓
Traefik :80/:443
 ↓
Let's Encrypt TLS
 ↓
Streamlit 127.0.0.1:8501
 ↓
FastAPI :8000 (Docker internal)
 ↓
FAISS + Neo4j (Docker internal)
```

---

# 25. Production Security

Public services:

```text
22   SSH
80   HTTP → HTTPS redirect
443  HTTPS
```

Localhost only:

```text
8501 Streamlit
```

Docker internal only:

```text
8000 FastAPI
7474 Neo4j HTTP
7687 Neo4j Bolt
```

FastAPI and Neo4j are not directly published to the public internet.

---

# 26. HTTPS and Reverse Proxy

Production HTTPS is handled by **Traefik** with **Let's Encrypt** certificate management.

Production URL:

```text
https://www.hybridrag.space
```

---

# 27. Docker Networking

The frontend communicates with the backend using:

```text
http://backend:8000
```

The backend communicates with Neo4j using:

```text
bolt://neo4j:7687
```

These service names are resolved internally through Docker networking.

---

# 28. Production Health Check

Check containers:

```bash
docker compose ps
```

Expected services:

```text
hybrid-kg-rag-frontend
hybrid-kg-rag-backend
hybrid-kg-rag-neo4j
```

Test production HTTPS:

```bash
curl -I https://www.hybridrag.space
```

A successful deployment should return a valid HTTPS response without TLS certificate errors.

---

# 29. Updating Production

```bash
cd /root/hybrid-knowledge-graph-rag
git pull origin main
docker compose build
docker compose up -d
docker compose ps
```

Persistent volumes preserve application state during normal container recreation.

---

# 30. Security Notes

Never commit:

```text
.env
API keys
Neo4j passwords
LangSmith keys
private credentials
runtime databases
uploaded private documents
```

If a secret is exposed:

1. Revoke or rotate it.
2. Update `.env`.
3. Recreate the affected container.
4. Verify application health.

---

# 31. RAG Decision Flow

```text
User Question
 ↓
Input Guardrail
 ↓
Conversation History
 ↓
Query Rewriting
 ↓
Hybrid Retrieval
 ├── FAISS
 └── Neo4j
 ↓
Fusion / Ranking
 ↓
Evidence Sufficiency
 ├── Sufficient → Document
 ├── Partial → Document + validated Web
 └── Insufficient → validated Web
 ↓
Context Builder
 ↓
LangChain Messages
 ↓
LLM
 ↓
Output Guardrail
 ↓
Final Response + Sources
```

---

# 32. Why Hybrid RAG?

Vector databases answer:

> Which chunks are semantically similar to this question?

Knowledge graphs answer:

> Which entities are connected and how are they related?

Combining both provides:

```text
Semantic Similarity
        +
Structured Relationships
        =
Hybrid Knowledge Retrieval
```

---

# 33. Cost-Control Strategy

The system reduces unnecessary LLM usage by:

- processing documents once
- persisting FAISS indexes
- persisting Neo4j graph data
- persisting conversation memory
- reusing stored document knowledge
- using bounded retrieval
- limiting context size
- searching the web only when required
- validating web evidence before generation

---

# 34. Current Production Status

```text
Domain             hybridrag.space
HTTPS              Enabled
TLS                 Let's Encrypt
Reverse Proxy       Traefik
Frontend            Streamlit
Backend             FastAPI
Vector Store        FAISS
Knowledge Graph     Neo4j
Memory              SQLite
Web Fallback        Enabled
Guardrails          NeMo Guardrails
Evaluation          DeepEval
Observability       LangSmith
Containers          Docker
Orchestration       Docker Compose
```

Production application:

**https://www.hybridrag.space**

---

# 35. Future Improvements

Potential future improvements:

- DOCX ingestion
- additional document formats
- graph visualization
- advanced reranking
- improved entity resolution
- richer citation rendering
- authentication
- user accounts
- role-based access control
- rate limiting
- background ingestion workers
- Redis caching
- asynchronous processing
- automated CI/CD
- automated backups
- retrieval regression testing
- monitoring and alerting
- multi-user conversation isolation
- object storage

---

# 36. Repository

https://github.com/rajkumarkathiravan18-lgtm/hybrid-knowledge-graph-rag

---

# 37. Author

**Raj Kumar**

**Production Hybrid Knowledge Graph RAG Application**

End-to-end implementation covering document ingestion, vector retrieval, knowledge graphs, conversational memory, web fallback, guardrails, evaluation, observability, containerization, and VPS deployment.

---

## License

Add the appropriate license before distributing or using this project in environments that require an explicit software license.
