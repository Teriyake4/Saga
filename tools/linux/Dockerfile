FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential clang libclang-rt-dev cmake ninja-build python3 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
