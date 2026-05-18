FROM python:3.11-slim

WORKDIR /app

# Runtime-only deps (no pytest/httpx). Local dev still uses requirements.txt via `make install`.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Pre-download the FastEmbed ONNX model into the image so first request
# doesn't pay a ~30 MB download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY . .

# Pre-encode the corpus into data/dense_index.npz. Cloud Run 1-CPU
# cold-start can't encode 497 snippets in <30s; baking moves that cost
# into the (beefier) Cloud Build step.
RUN python scripts/build_dense_index.py

# Cloud Run sets PORT=8080. docker-compose / local override sets PORT=8000.
ENV PORT=8080
ENV PYTHONPATH=/app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT', '8080')}/health\")" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
