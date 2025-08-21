FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for numpy/scipy if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# default envs (override at deploy)
ENV ARTIFACT_DIR=artifacts
ENV LABEL_TYPE=log_return
EXPOSE 8080

# use multiple workers only if your host has CPU to spare
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
