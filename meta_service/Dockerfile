FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV HOST=0.0.0.0

RUN apt-get update && apt-get install -y \
    gcc g++ libc6-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser -m

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ðŸ‘‡ CRITICAL FIX: Using uvicorn directly so it doesn't crash
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]
