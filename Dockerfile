FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY src/ src/
RUN pip install --no-cache-dir .

FROM python:3.13-slim
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/sentrysloth /usr/local/bin/sentrysloth
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash sentrysloth
USER sentrysloth
WORKDIR /home/sentrysloth
ENTRYPOINT ["sentrysloth"]
