"""CPU-only tests for the Phase 1 gRPC benchmark/conformance harness."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.grpc.clients import (  # noqa: E402
    NativeGrpcClient,
    RequestResult,
    RequestSpec,
    StreamObservation,
)
from benchmark.grpc.conformance import (  # noqa: E402
    _compare_authoritative_tokens,
    _first_difference_offset,
    classify_repeatability,
    compare_results,
)
from benchmark.grpc.harness import run_workload, write_jsonl  # noqa: E402
from benchmark.grpc.metrics import summarize  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class _AsyncStream:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        self.iterator = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


class _Stub:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def TextGenerate(self, request, timeout):
        self.requests.append((request, timeout))
        return _AsyncStream(self.responses)


class _Pb2:
    class SamplingParams(SimpleNamespace):
        pass

    class TextGenerateRequest(SimpleNamespace):
        pass


def _response(text, tokens, *, finished=False, finish_reason=None):
    metadata = {"completion_tokens": str(tokens)}
    if finish_reason:
        metadata["finish_reason"] = finish_reason
    return SimpleNamespace(text=text, meta_info=metadata, finished=finished)


def _result(protocol, spec, text="ABC", tokens=3, *, finished=True):
    return RequestResult(
        protocol=protocol,
        request=spec,
        success=True,
        generated_text=text,
        output_len=tokens,
        finished=finished,
    )


def _comparison(divergent_indexes=()):
    checks = [
        {
            "request_index": index,
            "conformant": index not in divergent_indexes,
            "first_divergence": None,
            "chunk_shape_difference": False,
            "token_level": {"status": "unavailable"},
        }
        for index in range(2)
    ]
    return {"checks": checks}


class TestGrpcBenchmarkHarness(CustomTestCase):
    def setUp(self):
        self.spec = RequestSpec(0, "hello", 1, 3, "template-rid")

    def test_native_stream_records_chunk_and_estimated_token_timing(self):
        responses = [
            _response("AB", 2),
            _response("ABCDEF", 6, finished=True, finish_reason="length"),
        ]
        stub = _Stub(responses)
        client = NativeGrpcClient(
            "localhost:1",
            {"temperature": 0.0, "ignore_eos": True},
            stub=stub,
            pb2=_Pb2,
        )
        ticks = iter([10.0, 10.1, 10.5, 10.6])
        with patch(
            "benchmark.grpc.clients.time.perf_counter",
            side_effect=lambda: next(ticks),
        ):
            result = asyncio.run(client.generate(self.spec, 9.0))
        self.assertTrue(result.success)
        self.assertEqual(result.generated_text, "ABCDEF")
        self.assertEqual(result.output_len, 6)
        self.assertAlmostEqual(result.ttft_s, 0.1)
        self.assertEqual(len(result.chunk_intervals_s), 1)
        self.assertAlmostEqual(result.chunk_intervals_s[0], 0.4)
        self.assertEqual(len(result.estimated_token_itls_s), 4)
        for value in result.estimated_token_itls_s:
            self.assertAlmostEqual(value, 0.1)
        self.assertEqual(
            [item.attributed_tokens for item in result.observations], [2, 4]
        )
        self.assertEqual(result.observations[0].text_delta_length, 2)
        self.assertEqual(result.observations[-1].finish_reason, "length")
        proto_request, _ = stub.requests[0]
        self.assertEqual(proto_request.rid, "template-rid")
        self.assertEqual(proto_request.sampling_params.max_new_tokens, 3)

    def test_cumulative_text_violation_is_recorded(self):
        client = NativeGrpcClient(
            "localhost:1",
            {},
            stub=_Stub([_response("ABC", 1), _response("XYZ", 2, finished=True)]),
            pb2=_Pb2,
        )
        result = asyncio.run(client.generate(self.spec, 0.0))
        self.assertTrue(result.cumulative_stream_violation)
        self.assertFalse(result.observations[-1].prefix_monotonic)
        peer = _result("http", self.spec, text="XYZ", tokens=2)
        report = compare_results([peer], [result])
        self.assertEqual(
            report["checks"][0]["first_divergence"]["kind"],
            "cumulative-stream-violation",
        )

    def test_metadata_before_text_does_not_zero_ttft(self):
        client = NativeGrpcClient(
            "localhost:1",
            {},
            stub=_Stub([_response("", 0), _response("A", 1, finished=True)]),
            pb2=_Pb2,
        )
        ticks = iter([10.0, 10.05, 10.2, 10.3])
        with patch(
            "benchmark.grpc.clients.time.perf_counter",
            side_effect=lambda: next(ticks),
        ):
            result = asyncio.run(client.generate(self.spec, 0.0))
        self.assertAlmostEqual(result.ttft_s, 0.2)
        self.assertEqual(len(result.chunk_intervals_s), 1)
        self.assertAlmostEqual(result.chunk_intervals_s[0], 0.15)
        self.assertEqual(result.estimated_token_itls_s, [])

    def test_empty_and_truncated_streams_are_failures(self):
        empty_client = NativeGrpcClient("localhost:1", {}, stub=_Stub([]), pb2=_Pb2)
        empty = asyncio.run(empty_client.generate(self.spec, 0.0))
        self.assertFalse(empty.success)
        self.assertIn("stream ended", empty.error)

        truncated_client = NativeGrpcClient(
            "localhost:1", {}, stub=_Stub([_response("partial", 1)]), pb2=_Pb2
        )
        truncated = asyncio.run(truncated_client.generate(self.spec, 0.0))
        self.assertFalse(truncated.success)
        self.assertEqual(truncated.generated_text, "partial")
        self.assertIn("explicit terminal response", truncated.error)

    def test_unknown_sampling_field_fails_before_transport(self):
        stub = _Stub([])
        client = NativeGrpcClient("localhost:1", {"made_up": 1}, stub=stub, pb2=_Pb2)
        result = asyncio.run(client.generate(self.spec, 0.0))
        self.assertFalse(result.success)
        self.assertIn("unsupported gRPC sampling fields", result.error)
        self.assertEqual(stub.requests, [])

    def test_first_character_divergence(self):
        self.assertEqual(_first_difference_offset("ABC", "XBC"), 0)
        self.assertEqual(_first_difference_offset("ABC", "ABX"), 2)
        self.assertEqual(_first_difference_offset("ABC", "ABCx"), 3)
        report = compare_results(
            [_result("http", self.spec, "ABC")],
            [_result("native-grpc", self.spec, "XBC")],
        )
        divergence = report["checks"][0]["first_divergence"]
        self.assertEqual(divergence["kind"], "generated-text-mismatch")
        self.assertEqual(divergence["first_differing_character_offset"], 0)

    def test_finish_reason_is_normalized_and_compared_when_available(self):
        client = NativeGrpcClient(
            "localhost:1",
            {},
            stub=_Stub([_response("ABC", 3, finished=True, finish_reason='"length"')]),
            pb2=_Pb2,
        )
        candidate = asyncio.run(client.generate(self.spec, 0.0))
        self.assertEqual(candidate.finish_reason, "length")
        baseline = _result("http", self.spec)
        baseline.finish_reason = "stop"
        divergence = compare_results([baseline], [candidate])["checks"][0][
            "first_divergence"
        ]
        self.assertEqual(divergence["kind"], "finish-reason-mismatch")

    def test_chunk_shape_is_diagnostic_not_semantic_failure(self):
        left = _result("http", self.spec)
        right = _result("native-grpc", self.spec)
        left.observations = [
            StreamObservation(0.1, 0.1, "a", "a", 1, 1, 1, False, None, True)
        ]
        right.observations = [
            StreamObservation(0.1, 0.1, "a", "a", 2, 1, 1, False, None, True)
        ]
        check = compare_results([left], [right])["checks"][0]
        self.assertTrue(check["conformant"])
        self.assertTrue(check["chunk_shape_difference"])

    def test_repeatability_classification(self):
        comparisons = {
            "http_a_vs_http_b": _comparison({0}),
            "grpc_a_vs_grpc_b": _comparison(),
            "http_a_vs_grpc_a": _comparison({0, 1}),
            "http_b_vs_grpc_b": _comparison({1}),
        }
        report = classify_repeatability(comparisons)
        self.assertEqual(report["counts"]["both"], 1)
        self.assertEqual(report["counts"]["cross-protocol-only-divergence"], 1)

    def test_repeated_executions_use_distinct_rids(self):
        class TrackingClient:
            protocol = "fake"

            def __init__(client_self):
                client_self.rids = []

            async def generate(client_self, request, run_start):
                client_self.rids.append(request.request_id)
                return _result("fake", request, "x", 1)

            async def close(client_self):
                pass

        client = TrackingClient()
        asyncio.run(run_workload(client, [self.spec], 1, warmup=0, execution_id="a"))
        asyncio.run(run_workload(client, [self.spec], 1, warmup=0, execution_id="b"))
        self.assertNotEqual(client.rids[0], client.rids[1])
        self.assertIn(self.spec.identity, client.rids[0])

    def test_token_level_comparison_is_explicitly_unavailable(self):
        status = _compare_authoritative_tokens(
            _result("http", self.spec), _result("native-grpc", self.spec)
        )
        self.assertEqual(status["status"], "unavailable")
        self.assertIn("authoritative output IDs", status["reason"])

    def test_summary_uses_explicit_latency_family_names(self):
        result = _result("http", self.spec)
        result.latency_s = 0.5
        result.ttft_s = 0.1
        result.chunk_intervals_s = [0.2]
        result.estimated_token_itls_s = [0.1, 0.1]
        result.observations = [
            StreamObservation(0.1, 0.1, "a", "a", 1, 1, 1, False, None, True),
            StreamObservation(0.3, 0.2, "b", "b", 2, 3, 2, True, "length", True),
        ]
        summary = summarize([result], 1.0)
        self.assertEqual(summary["chunk_interval_ms"]["p50"], 200)
        self.assertEqual(summary["estimated_token_itl_ms"]["p50"], 100)
        self.assertNotIn("itl_ms", summary)
        self.assertEqual(summary["chunk_shape"]["maximum_chunk_tokens"], 2)

    def test_schema_v2_artifact_keeps_v1_itl_alias(self):
        result = _result("http", self.spec)
        result.estimated_token_itls_s = [0.1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            write_jsonl(path, [result.artifact("run")])
            record = json.loads(path.read_text())
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["itls_s"], record["estimated_token_itls_s"])
        self.assertEqual(record["request_identity"], self.spec.identity)
        self.assertNotIn("prompt", record)


if __name__ == "__main__":
    unittest.main()
