ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project metadata first to leverage Docker cache for dependency installation
COPY pyproject.toml README.md /workspace/
COPY . /workspace

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# Non-root user
RUN groupadd -g ${GID} sam && useradd -m -u ${UID} -g ${GID} sam || true
USER sam

ENV PYTHONUNBUFFERED=1

CMD ["python", "eval/main.py"]
