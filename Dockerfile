FROM docker.io/library/gcc:14 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG BAZEL_VERSION=7.6.0
RUN wget -q "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64" \
    -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

WORKDIR /app
COPY . .

RUN bazel build //src:rinha --config=release

FROM docker.io/library/debian:trixie-slim AS runtime

COPY --from=builder /app/bazel-bin/src/rinha /usr/local/bin/rinha

CMD ["rinha"]
