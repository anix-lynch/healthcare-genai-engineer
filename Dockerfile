FROM python:3.11-slim

WORKDIR /app

# Runtime-only deps (no pytest/httpx). Local dev still uses requirements.txt via `make install`.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Cloud Run sets PORT=8080. docker-compose / local override sets PORT=8000.
ENV PORT=8080
ENV PYTHONPATH=/app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT', '8080')}/health\")" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
