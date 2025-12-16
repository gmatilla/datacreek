FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY setup.py ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir .
COPY datacreek ./datacreek
COPY configs ./configs
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "datacreek.api:app", "--host", "0.0.0.0", "--port", "8000"]
