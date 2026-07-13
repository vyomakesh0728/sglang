"""Latency and stream-shape aggregation for transport benchmarks."""

from __future__ import annotations

import math
import statistics
from typing import Any

from benchmark.grpc.clients import RequestResult


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, Any]:
    valid = [result for result in results if result.success]
    ttfts = [result.ttft_s * 1000 for result in valid]
    e2e = [result.latency_s * 1000 for result in valid]
    chunk_intervals = [
        value * 1000 for result in valid for value in result.chunk_intervals_s
    ]
    estimated_itls = [
        value * 1000 for result in valid for value in result.estimated_token_itls_s
    ]
    output_tokens = sum(result.output_len for result in valid)
    chunks_per_request = [len(result.observations) for result in valid]
    tokens_per_chunk = [
        observation.attributed_tokens
        for result in valid
        for observation in result.observations
    ]
    first_chunk_tokens = [
        result.observations[0].attributed_tokens
        for result in valid
        if result.observations
    ]
    return {
        "requests": len(results),
        "completed": len(valid),
        "failures": len(results) - len(valid),
        "duration_s": duration_s,
        "request_throughput": len(valid) / duration_s if duration_s else 0,
        "output_throughput": output_tokens / duration_s if duration_s else 0,
        "ttft_ms": _percentiles(ttfts),
        "chunk_interval_ms": _percentiles(chunk_intervals),
        "estimated_token_itl_ms": _percentiles(estimated_itls),
        "e2e_ms": _percentiles(e2e),
        "chunk_shape": {
            "chunks_per_request": _percentiles(chunks_per_request),
            "tokens_per_chunk": _percentiles(tokens_per_chunk),
            "first_chunk_tokens": _percentiles(first_chunk_tokens),
            "maximum_chunk_tokens": max(tokens_per_chunk, default=0),
        },
        "cumulative_stream_violations": sum(
            result.cumulative_stream_violation for result in results
        ),
        "errors": [result.error for result in results if result.error],
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99")}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }
