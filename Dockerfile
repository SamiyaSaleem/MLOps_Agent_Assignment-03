FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.joblib

COPY requirements-inference.txt .

RUN pip install --no-cache-dir -r requirements-inference.txt

COPY api/ ./api/
COPY models/model.joblib ./models/model.joblib

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
