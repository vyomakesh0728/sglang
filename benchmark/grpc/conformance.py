"""Semantic conformance and same-protocol repeatability analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.grpc.clients import RequestResult


def compare_results(
    baseline: list[RequestResult], candidate: list[RequestResult]
) -> dict[str, Any]:
    left = {item.request.index: item for item in baseline}
    right = {item.request.index: item for item in candidate}
    indexes = sorted(set(left) | set(right))
    checks = []
    for index in indexes:
        a, b = left.get(index), right.get(index)
        semantic_divergence = _first_semantic_divergence(a, b)
        chunk_shape_difference = (
            _chunk_shape(a) != _chunk_shape(b) if a and b else False
        )
        token_level = _compare_authoritative_tokens(a, b)
        checks.append(
            {
                "request_index": index,
                "conformant": semantic_divergence is None,
                "first_divergence": semantic_divergence,
                "chunk_shape_difference": chunk_shape_difference,
                "token_level": token_level,
            }
        )
    conformant = sum(item["conformant"] for item in checks)
    divergence_counts = dict(
        Counter(
            item["first_divergence"]["kind"]
            for item in checks
            if item["first_divergence"] is not None
        )
    )
    return {
        "requests": len(indexes),
        "conformant": conformant,
        "conformance_rate": conformant / len(indexes) if indexes else 1.0,
        "first_divergence_counts": divergence_counts,
        "chunk_shape_differences": sum(
            item["chunk_shape_difference"] for item in checks
        ),
        "token_level_status": "unavailable"
        if all(item["token_level"]["status"] == "unavailable" for item in checks)
        else "available",
        "checks": checks,
    }


def classify_repeatability(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_name = {
        name: {
            item["request_index"]: not item["conformant"] for item in report["checks"]
        }
        for name, report in comparisons.items()
    }
    indexes = sorted({index for values in by_name.values() for index in values})
    classifications = []
    for index in indexes:
        same = by_name["http_a_vs_http_b"].get(index, False) or by_name[
            "grpc_a_vs_grpc_b"
        ].get(index, False)
        cross = by_name["http_a_vs_grpc_a"].get(index, False) or by_name[
            "http_b_vs_grpc_b"
        ].get(index, False)
        if same and cross:
            classification = "both"
        elif same:
            classification = "same-protocol-nondeterminism"
        elif cross:
            classification = "cross-protocol-only-divergence"
        else:
            classification = "stable-and-conformant"
        classifications.append(
            {"request_index": index, "classification": classification}
        )
    counts = dict(Counter(item["classification"] for item in classifications))
    return {"counts": counts, "requests": classifications}


def _first_semantic_divergence(
    baseline: RequestResult | None, candidate: RequestResult | None
) -> dict[str, Any] | None:
    if baseline is None or candidate is None:
        return {"kind": "missing-result"}
    if not baseline.success or not candidate.success:
        return {
            "kind": "request-failure",
            "baseline_error": baseline.error,
            "candidate_error": candidate.error,
        }
    if baseline.generated_text != candidate.generated_text:
        offset = _first_difference_offset(
            baseline.generated_text, candidate.generated_text
        )
        return {
            "kind": "generated-text-mismatch",
            "first_differing_character_offset": offset,
            "baseline_length": len(baseline.generated_text),
            "candidate_length": len(candidate.generated_text),
        }
    if baseline.output_len != candidate.output_len:
        return {
            "kind": "output-length-mismatch",
            "baseline": baseline.output_len,
            "candidate": candidate.output_len,
        }
    if baseline.finished != candidate.finished:
        return {
            "kind": "terminal-state-mismatch",
            "baseline": baseline.finished,
            "candidate": candidate.finished,
        }
    if baseline.cumulative_stream_violation or candidate.cumulative_stream_violation:
        return {
            "kind": "cumulative-stream-violation",
            "baseline": baseline.cumulative_stream_violation,
            "candidate": candidate.cumulative_stream_violation,
        }
    return None


def _first_difference_offset(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _chunk_shape(result: RequestResult) -> list[tuple[int, int, bool]]:
    return [
        (
            item.text_delta_length,
            item.attributed_tokens,
            item.finished,
        )
        for item in result.observations
    ]


def _compare_authoritative_tokens(
    baseline: RequestResult | None, candidate: RequestResult | None
) -> dict[str, Any]:
    if (
        baseline is None
        or candidate is None
        or baseline.authoritative_output_ids is None
        or candidate.authoritative_output_ids is None
    ):
        return {
            "status": "unavailable",
            "reason": "TextGenerate transports do not expose authoritative output IDs",
        }
    return {
        "status": "equal"
        if baseline.authoritative_output_ids == candidate.authoritative_output_ids
        else "different"
    }
