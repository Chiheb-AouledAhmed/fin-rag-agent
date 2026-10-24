FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && useradd --create-home app
USER app

ENV FINRAG_CORPUS=/app/data/corpus \
    FINRAG_PROVIDER=offline
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "finrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
