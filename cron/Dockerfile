FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3

COPY cron/cleanup_checkpoints.py ./cleanup_checkpoints.py

ENTRYPOINT ["python", "/app/cleanup_checkpoints.py"]
