FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      git \
      python3-dev \
      libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/TorchDSP/torchsig

COPY . .

CMD ["bash"]
