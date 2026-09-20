# ============================================================
# Hybrid Knowledge Graph RAG
# Frontend Docker Image
# ============================================================

FROM python:3.11-slim


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

# Prevent Python from creating .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to Docker logs.
ENV PYTHONUNBUFFERED=1


# ============================================================
# APPLICATION DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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

COPY app.py /app/app.py
COPY api_client.py /app/api_client.py


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


# ============================================================
# STREAMLIT PORT
# ============================================================

EXPOSE 8501


# ============================================================
# START STREAMLIT
# ============================================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]