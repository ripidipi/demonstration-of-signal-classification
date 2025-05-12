FROM --platform=linux/arm64 python:3.10-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      git \
      python3-dev \
      libatlas-base-dev && \
      rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/TorchDSP/torchsig

FROM python:3.10-slim-bookworm
WORKDIR /app

RUN useradd --system --no-create-home appuser
COPY --from=builder /wheels /wheels
COPY requirements.txt . 

RUN pip install --no-index --find-links=/wheels -r requirements.txt

USER appuser

COPY --chown=appuser:appuser src/ ./src
CMD ["python", "src/train.py"]
