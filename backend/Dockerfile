# ============================================================
# Hybrid Knowledge Graph RAG
# Backend Docker Image
# ============================================================

FROM python:3.11-slim

# Prevent Python from generating .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to Docker logs.
ENV PYTHONUNBUFFERED=1

# Application working directory.
WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        -r /app/requirements.txt


# ============================================================
# APPLICATION SOURCE
# ============================================================

COPY app /app/app


# ============================================================
# RUNTIME DIRECTORIES
# ============================================================

RUN mkdir -p \
    /app/data/uploads \
    /app/data/faiss \
    /app/data/memory \
    /app/logs


# ============================================================
# FASTAPI PORT
# ============================================================

EXPOSE 8000


# ============================================================
# START FASTAPI
# ============================================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]