FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts never download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY cli.py .

EXPOSE 8000
# Run migrations, then serve. Render injects $PORT.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
