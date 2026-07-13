# Native gRPC Phase 1 benchmark and conformance harness

This diagnostic harness sends an identical ordered workload to HTTP `/generate`
and native `SglangService.TextGenerate`. It separates final-output semantics,
stream shape, and timing so transport differences are not confused with runtime
nondeterminism.

Python protobuf stubs are generated from
`proto/sglang/runtime/v1/sglang.proto` in a temporary directory. Generated
bindings are never committed.

The implementation is split by review boundary:

- `harness.py`: workload orchestration, artifacts, metadata, and CLI.
- `clients.py`: HTTP/native-gRPC clients and stream observations.
- `metrics.py`: latency and chunk-shape aggregation.
- `conformance.py`: semantic divergence and repeatability classification.

The harness intentionally does not call `sglang.benchmark.serving.calculate_metrics`:
that function requires tokenizer-driven serving structures and reports inferred values
as ITL. This harness must remain standalone and preserve the distinction between
observed chunk intervals and estimated token-level ITL.

## Setup

Build/install the current checkout, including the `sglang-grpc` Rust extension:

```bash
pip install grpcio grpcio-tools
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 30000 --grpc-port 30001
```

## Cross-protocol run

```bash
python benchmark/grpc/harness.py \
  --http-url http://127.0.0.1:30000 \
  --grpc-target 127.0.0.1:30001 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 32 --input-len 128 --output-len 32 \
  --concurrency 8 --warmup 2 \
  --flush-cache-url http://127.0.0.1:30000 \
  --output grpc-benchmark-results
```

`--prompts prompts.jsonl` accepts one `{"prompt": "..."}` record per line.
Synthetic `--input-len` is a word count, not a tokenizer token count. Unless a
future client is given the model tokenizer, actual input-token distribution is
reported as unavailable rather than guessed.

## Repeatability diagnosis

Before interpreting cross-protocol mismatches, run:

```bash
python benchmark/grpc/harness.py \
  --repeatability \
  --http-url http://127.0.0.1:30000 \
  --grpc-target 127.0.0.1:30001 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 32 --input-len 128 --output-len 32 \
  --concurrency 32 --warmup 2 \
  --flush-cache-url http://127.0.0.1:30000 \
  --output grpc-repeatability-c32
```

This executes and compares:

- HTTP A vs HTTP B
- native gRPC A vs native gRPC B
- HTTP A vs native gRPC A
- HTTP B vs native gRPC B

The prompt hash plus index forms a stable request identity. Every execution gets
a distinct wire RID, preventing stale state from making repeated runs collide.
Requests are classified as:

- `stable-and-conformant`
- `same-protocol-nondeterminism`
- `cross-protocol-only-divergence`
- `both`

## Artifact fields

The output directory contains one `<execution>.requests.jsonl` file and a
`comparison.json`. Schema version 2 request records contain no raw prompt, only
its SHA-256 hash. Generated text remains available to diagnose final-output
mismatches.

Each ordered `stream_observations` entry records:

- arrival offset and observed chunk interval;
- cumulative-text and delta-text hashes;
- delta character length;
- reported completion-token count and tokens attributed to the chunk;
- terminal flag and finish reason, when available;
- whether cumulative text remained prefix-monotonic.

The legacy `itls_s` request field remains as an alias for schema-v1 readers, but
new analysis must use the explicit metric names below.

### Timing definitions

- `ttft_ms`: request start to first non-empty text observation.
- `chunk_interval_ms`: directly observed interval between stream observations.
- `estimated_token_itl_ms`: chunk interval divided equally among newly reported
  completion tokens. It is an estimate, not observed per-token arrival latency.
- `e2e_ms`: request start through clean stream termination.

Chunk summaries include chunks per request, tokens per chunk, first-chunk token
count, and maximum chunk size. A chunk-shape difference is diagnostic only: it
does not fail semantic conformance when final output, output length, and terminal
state agree.

## Divergence interpretation

For each paired request, conformance reports the first concrete semantic
failure: missing result, request failure, generated-text mismatch and first
character offset, output-length mismatch, terminal-state mismatch, finish-reason
mismatch when both transports report one, or a cumulative-stream violation.
Chunk-shape differences are reported separately.

HTTP and native `TextGenerate` return text, not authoritative output token IDs.
Token-level comparison is therefore explicitly `unavailable`. The harness does
not retokenize decoded text and call that token equality.

Performance claims remain **diagnostic** until same-protocol repeatability and
cross-protocol semantic behavior are understood. Cancellation and slow-consumer
experiments should remain separate workloads so artificial consumer delay does
not pollute latency metrics. Legacy `smg-grpc-servicer` uses a different external
proto and remains out of scope.
