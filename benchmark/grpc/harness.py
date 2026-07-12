#!/usr/bin/env python3
"""Benchmark HTTP /generate and native TextGenerate with one workload.

The harness intentionally lives outside the production client package. Native Python
stubs are generated from the repository's canonical proto into a temporary directory.
This keeps generated code out of the tree and makes proto drift visible immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_ROOT = REPO_ROOT / "proto"
PROTO_FILE = PROTO_ROOT / "sglang/runtime/v1/sglang.proto"


@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt: str
    prompt_len: int
    output_len: int
    request_id: str

    @property
    def identity(self) -> str:
        return f"{self.index}-{_sha256(self.prompt)[:16]}"

    def for_execution(self, execution_id: str) -> "RequestSpec":
        return RequestSpec(
            index=self.index,
            prompt=self.prompt,
            prompt_len=self.prompt_len,
            output_len=self.output_len,
            request_id=f"bench-{self.identity}-{execution_id}",
        )


@dataclass
class StreamObservation:
    arrival_offset_s: float
    chunk_interval_s: float
    cumulative_text_sha256: str
    text_delta_sha256: str
    text_delta_length: int
    reported_completion_tokens: int
    attributed_tokens: int
    finished: bool
    finish_reason: str | None
    prefix_monotonic: bool

    def artifact(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestResult:
    protocol: str
    request: RequestSpec
    success: bool = False
    generated_text: str = ""
    output_len: int = 0
    latency_s: float = 0.0
    ttft_s: float = 0.0
    estimated_token_itls_s: list[float] = field(default_factory=list)
    chunk_intervals_s: list[float] = field(default_factory=list)
    observations: list[StreamObservation] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None
    error: str = ""
    start_offset_s: float = 0.0
    authoritative_output_ids: list[int] | None = None

    @property
    def itls_s(self) -> list[float]:
        """Backward-readable alias for schema-v1 consumers."""
        return self.estimated_token_itls_s

    @property
    def cumulative_stream_violation(self) -> bool:
        return any(not item.prefix_monotonic for item in self.observations)

    def artifact(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "protocol": self.protocol,
            "request_index": self.request.index,
            "request_identity": self.request.identity,
            "request_id": self.request.request_id,
            "prompt_sha256": _sha256(self.request.prompt),
            "prompt_len": self.request.prompt_len,
            "requested_output_len": self.request.output_len,
            "success": self.success,
            "generated_text": self.generated_text,
            "generated_text_sha256": _sha256(self.generated_text),
            "reported_output_len": self.output_len,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "chunk_intervals_s": self.chunk_intervals_s,
            "estimated_token_itls_s": self.estimated_token_itls_s,
            "itls_s": self.estimated_token_itls_s,
            "stream_observations": [item.artifact() for item in self.observations],
            "cumulative_stream_violation": self.cumulative_stream_violation,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "start_offset_s": self.start_offset_s,
            "authoritative_output_ids": self.authoritative_output_ids,
        }


class Client(Protocol):
    protocol: str

    async def generate(
        self, request: RequestSpec, run_start: float
    ) -> RequestResult: ...

    async def close(self) -> None: ...


class HttpGenerateClient:
    protocol = "http"

    def __init__(self, url: str, sampling: dict[str, Any], timeout_s: float = 3600):
        self.url = url.rstrip("/") + "/generate"
        self.sampling = sampling
        self.timeout_s = timeout_s
        self._session = None

    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp
            except ImportError as exc:
                raise RuntimeError("HTTP benchmark requires aiohttp") from exc
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def generate(self, request: RequestSpec, run_start: float) -> RequestResult:
        result = RequestResult(protocol=self.protocol, request=request)
        started = time.perf_counter()
        result.start_offset_s = started - run_start
        most_recent = started
        completion_tokens = 0
        previous_text = ""
        payload = {
            "text": request.prompt,
            "sampling_params": {**self.sampling, "max_new_tokens": request.output_len},
            "stream": True,
            "rid": request.request_id,
        }
        try:
            session = await self._get_session()
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                    return result
                async for raw in response.content:
                    body = raw.strip()
                    if not body:
                        continue
                    if body.startswith(b"data:"):
                        body = body[5:].strip()
                    if not body or body == b"[DONE]":
                        continue
                    data = json.loads(body)
                    text = data.get("text", "")
                    meta = data.get("meta_info") or {}
                    token_count = _int_or_zero(meta.get("completion_tokens"))
                    now = time.perf_counter()
                    chunk_finished = bool(data.get("finished", False))
                    finish_reason = data.get("finish_reason") or meta.get(
                        "finish_reason"
                    )
                    _record_stream_observation(
                        result,
                        now=now,
                        started=started,
                        most_recent=most_recent,
                        previous_text=previous_text,
                        current_text=text,
                        previous_tokens=completion_tokens,
                        reported_tokens=token_count,
                        finished=chunk_finished,
                        finish_reason=finish_reason,
                    )
                    most_recent = now
                    previous_text = text
                    completion_tokens = max(completion_tokens, token_count)
                    result.finished = chunk_finished or result.finished
                    result.finish_reason = finish_reason or result.finish_reason
            result.generated_text = previous_text
            # A clean HTTP stream EOF (or [DONE]) is the transport's terminal signal.
            result.finished = result.finished or bool(previous_text)
            result.output_len = completion_tokens
            result.latency_s = time.perf_counter() - started
            result.success = result.finished or bool(previous_text)
            if not result.success:
                result.error = "stream ended without output or terminal response"
        except asyncio.CancelledError:
            result.error = "cancelled"
            result.latency_s = time.perf_counter() - started
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_s = time.perf_counter() - started
        return result


class NativeGrpcClient:
    protocol = "native-grpc"

    def __init__(
        self,
        target: str,
        sampling: dict[str, Any],
        timeout_s: float = 3600,
        *,
        stub: Any = None,
        pb2: Any = None,
    ):
        self.target = _grpc_target(target)
        self.sampling = sampling
        self.timeout_s = timeout_s
        self._stub = stub
        self._pb2 = pb2
        self._channel = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

    async def _initialize(self) -> None:
        if self._stub is not None:
            return
        try:
            import grpc
        except ImportError as exc:
            raise RuntimeError("native gRPC benchmark requires grpcio") from exc
        self._tempdir, self._pb2, pb2_grpc = compile_proto_stubs()
        self._channel = grpc.aio.insecure_channel(self.target)
        self._stub = pb2_grpc.SglangServiceStub(self._channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _build_request(self, request: RequestSpec):
        if self._pb2 is None:
            raise RuntimeError("gRPC client is not initialized")
        supported = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "min_new_tokens",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "n",
        }
        unknown = set(self.sampling) - supported
        if unknown:
            raise ValueError(f"unsupported gRPC sampling fields: {sorted(unknown)}")
        params = self._pb2.SamplingParams(
            **self.sampling, max_new_tokens=request.output_len
        )
        return self._pb2.TextGenerateRequest(
            text=request.prompt,
            sampling_params=params,
            stream=True,
            rid=request.request_id,
        )

    async def generate(self, request: RequestSpec, run_start: float) -> RequestResult:
        result = RequestResult(protocol=self.protocol, request=request)
        started = time.perf_counter()
        result.start_offset_s = started - run_start
        most_recent = started
        previous_text = ""
        completion_tokens = 0
        try:
            await self._initialize()
            call = self._stub.TextGenerate(
                self._build_request(request), timeout=self.timeout_s
            )
            async for response in call:
                now = time.perf_counter()
                text = response.text
                token_count = _int_or_zero(response.meta_info.get("completion_tokens"))
                finish_reason = response.meta_info.get("finish_reason")
                _record_stream_observation(
                    result,
                    now=now,
                    started=started,
                    most_recent=most_recent,
                    previous_text=previous_text,
                    current_text=text,
                    previous_tokens=completion_tokens,
                    reported_tokens=token_count,
                    finished=bool(response.finished),
                    finish_reason=finish_reason,
                )
                most_recent = now
                previous_text = text
                completion_tokens = max(completion_tokens, token_count)
                result.finished = bool(response.finished) or result.finished
                result.finish_reason = finish_reason or result.finish_reason
            result.generated_text = previous_text
            result.output_len = completion_tokens
            result.latency_s = time.perf_counter() - started
            result.success = result.finished or bool(previous_text)
            if not result.success:
                result.error = "stream ended without output or terminal response"
        except asyncio.CancelledError:
            result.error = "cancelled"
            result.latency_s = time.perf_counter() - started
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_s = time.perf_counter() - started
        return result


def compile_proto_stubs():
    """Compile canonical proto into a temporary directory and import its stubs."""
    try:
        import grpc_tools  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "stub generation requires grpcio-tools; install it with `pip install grpcio-tools`"
        ) from exc
    tempdir = tempfile.TemporaryDirectory(prefix="sglang-grpc-stubs-")
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"--python_out={tempdir.name}",
        f"--grpc_python_out={tempdir.name}",
        str(PROTO_FILE),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode:
        tempdir.cleanup()
        raise RuntimeError(f"protoc failed: {proc.stderr.strip()}")
    sys.path.insert(0, tempdir.name)
    existing_sglang = sys.modules.get("sglang")
    generated_package = str(Path(tempdir.name) / "sglang")
    extended_package_path = False
    if existing_sglang is not None and hasattr(existing_sglang, "__path__"):
        existing_sglang.__path__.append(generated_package)
        extended_package_path = True
    try:
        pb2 = importlib.import_module("sglang.runtime.v1.sglang_pb2")
        pb2_grpc = importlib.import_module("sglang.runtime.v1.sglang_pb2_grpc")
    except Exception:
        tempdir.cleanup()
        raise
    finally:
        if extended_package_path:
            existing_sglang.__path__.remove(generated_package)
        sys.path.remove(tempdir.name)
    return tempdir, pb2, pb2_grpc


async def run_workload(
    client: Client,
    requests: list[RequestSpec],
    concurrency: int,
    warmup: int = 1,
    execution_id: str = "run",
) -> tuple[list[RequestResult], float]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if requests and warmup:
        warmup_results = await asyncio.gather(
            *(
                client.generate(
                    requests[i % len(requests)].for_execution(
                        f"{execution_id}-warmup-{i}"
                    ),
                    time.perf_counter(),
                )
                for i in range(warmup)
            )
        )
        if not any(item.success for item in warmup_results):
            raise RuntimeError(f"warmup failed: {warmup_results[0].error}")
    semaphore = asyncio.Semaphore(concurrency)
    run_start = time.perf_counter()

    async def one(spec: RequestSpec):
        async with semaphore:
            return await client.generate(spec.for_execution(execution_id), run_start)

    results = await asyncio.gather(*(one(spec) for spec in requests))
    return results, time.perf_counter() - run_start


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def load_requests(
    path: Path | None,
    count: int,
    input_len: int,
    output_len: int,
    seed: int,
) -> list[RequestSpec]:
    rng = random.Random(seed)
    if path:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        prompts = [record["prompt"] for record in records]
    else:
        words = ["benchmark", "transport", "scheduler", "token", "latency", "stream"]
        prompts = [
            " ".join(rng.choice(words) for _ in range(input_len)) for _ in range(count)
        ]
    return [
        RequestSpec(
            index=index,
            prompt=prompt,
            prompt_len=len(prompt.split()),
            output_len=output_len,
            request_id=f"bench-{index}-{uuid.uuid5(uuid.NAMESPACE_URL, prompt).hex[:12]}",
        )
        for index, prompt in enumerate(prompts[:count])
    ]


def _record_stream_observation(
    result: RequestResult,
    *,
    now: float,
    started: float,
    most_recent: float,
    previous_text: str,
    current_text: str,
    previous_tokens: int,
    reported_tokens: int,
    finished: bool,
    finish_reason: str | None,
) -> None:
    prefix_monotonic = current_text.startswith(previous_text)
    delta = _text_delta(previous_text, current_text)
    attributed_tokens = max(reported_tokens - previous_tokens, 1 if delta else 0)
    interval = now - most_recent
    if not result.observations:
        if delta:
            result.ttft_s = now - started
        estimated_tokens = 0
    else:
        result.chunk_intervals_s.append(interval)
        estimated_tokens = attributed_tokens
    if estimated_tokens:
        result.estimated_token_itls_s.extend(
            [interval / attributed_tokens] * estimated_tokens
        )
    result.observations.append(
        StreamObservation(
            arrival_offset_s=now - started,
            chunk_interval_s=interval,
            cumulative_text_sha256=_sha256(current_text),
            text_delta_sha256=_sha256(delta),
            text_delta_length=len(delta),
            reported_completion_tokens=reported_tokens,
            attributed_tokens=attributed_tokens,
            finished=finished,
            finish_reason=finish_reason,
            prefix_monotonic=prefix_monotonic,
        )
    )


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


def _text_delta(previous: str, current: str) -> str:
    if current.startswith(previous):
        return current[len(previous) :]
    # Native contracts are cumulative. Returning current makes a violation visible
    # in generated-text conformance instead of silently dropping the chunk.
    return current


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _grpc_target(value: str) -> str:
    value = (
        value.removeprefix("grpc://").removeprefix("http://").removeprefix("https://")
    )
    return value.rstrip("/")


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


async def _main(args: argparse.Namespace) -> int:
    requests = load_requests(
        args.prompts,
        args.num_prompts,
        args.input_len,
        args.output_len,
        args.seed,
    )
    sampling = {"temperature": args.temperature, "ignore_eos": args.ignore_eos}
    factories: dict[str, Callable[[], Client]] = {
        "http": lambda: HttpGenerateClient(args.http_url, sampling, args.timeout),
        "native-grpc": lambda: NativeGrpcClient(
            args.grpc_target, sampling, args.timeout
        ),
    }
    protocols = args.protocols.split(",")
    unknown = set(protocols) - factories.keys()
    if unknown:
        raise ValueError(f"unknown protocols: {sorted(unknown)}")
    run_id = uuid.uuid4().hex
    execution_protocols = (
        [
            ("http-a", "http"),
            ("http-b", "http"),
            ("grpc-a", "native-grpc"),
            ("grpc-b", "native-grpc"),
        ]
        if args.repeatability
        else [(protocol, protocol) for protocol in protocols]
    )
    all_results: dict[str, list[RequestResult]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for execution, protocol in execution_protocols:
        if args.flush_cache_url:
            _flush_cache(args.flush_cache_url)
            await asyncio.sleep(args.post_flush_delay)
        client = factories[protocol]()
        try:
            results, duration = await run_workload(
                client,
                requests,
                args.concurrency,
                args.warmup,
                execution_id=f"{run_id[:8]}-{execution}",
            )
        finally:
            await client.close()
        all_results[execution] = results
        summaries[execution] = summarize(results, duration)
        write_jsonl(
            args.output / f"{execution}.requests.jsonl",
            (item.artifact(run_id) for item in results),
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "metadata": _reproducibility_metadata(
            args, protocols, execution_protocols, sampling, requests
        ),
        "summaries": summaries,
        "token_level_comparison": {
            "status": "unavailable",
            "reason": "HTTP and native TextGenerate do not expose authoritative output IDs",
        },
    }
    if args.repeatability:
        comparisons = {
            "http_a_vs_http_b": compare_results(
                all_results["http-a"], all_results["http-b"]
            ),
            "grpc_a_vs_grpc_b": compare_results(
                all_results["grpc-a"], all_results["grpc-b"]
            ),
            "http_a_vs_grpc_a": compare_results(
                all_results["http-a"], all_results["grpc-a"]
            ),
            "http_b_vs_grpc_b": compare_results(
                all_results["http-b"], all_results["grpc-b"]
            ),
        }
        report["comparisons"] = comparisons
        report["classification"] = classify_repeatability(comparisons)
        conformance_rates = [item["conformance_rate"] for item in comparisons.values()]
    elif len(execution_protocols) == 2:
        report["conformance"] = compare_results(
            all_results[execution_protocols[0][0]],
            all_results[execution_protocols[1][0]],
        )
        conformance_rates = [report["conformance"]["conformance_rate"]]
    else:
        conformance_rates = [1.0]
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 1 if args.require_conformance and min(conformance_rates) < 1 else 0


def _reproducibility_metadata(
    args: argparse.Namespace,
    protocols: list[str],
    executions: list[tuple[str, str]],
    sampling: dict[str, Any],
    requests: list[RequestSpec],
) -> dict[str, Any]:
    versions = {}
    for package in ("aiohttp", "grpcio", "grpcio-tools", "protobuf", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "commit": args.commit or _git_commit(),
        "model": args.model,
        "image_identifier": args.image_identifier,
        "dependency_versions": versions,
        "protocols": protocols,
        "execution_order": [name for name, _ in executions],
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "cache_flush": {
            "enabled": bool(args.flush_cache_url),
            "url": args.flush_cache_url,
            "post_flush_delay_s": args.post_flush_delay,
        },
        "sampling": sampling,
        "requested_input_length": args.input_len,
        "requested_output_length": args.output_len,
        "num_prompts": len(requests),
        "input_token_distribution": {
            "status": "unavailable",
            "reason": "standalone harness was not given a tokenizer",
            "prompt_length_measurement": "whitespace-separated words",
            "observed": _percentiles([request.prompt_len for request in requests]),
        },
        "timing_definitions": {
            "ttft_ms": "request start to first non-empty cumulative text observation",
            "chunk_interval_ms": "wall-clock interval between adjacent stream observations",
            "estimated_token_itl_ms": (
                "chunk interval divided equally across newly reported completion "
                "tokens; not directly observed token arrivals"
            ),
            "e2e_ms": "request start through clean stream termination",
        },
        "repeatability_mode": args.repeatability,
    }


def _flush_cache(base_url: str) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/flush_cache", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"flush cache returned HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(f"failed to flush server cache: {exc}") from exc


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default="http,native-grpc")
    parser.add_argument(
        "--repeatability",
        action="store_true",
        help="run HTTP A/B and native-gRPC A/B and classify divergence",
    )
    parser.add_argument("--http-url", default="http://127.0.0.1:30000")
    parser.add_argument("--grpc-target", default="127.0.0.1:30001")
    parser.add_argument("--prompts", type=Path, help="JSONL with a prompt field")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--commit", help="source commit when .git is unavailable")
    parser.add_argument(
        "--image-identifier",
        default=os.environ.get("SGLANG_BENCHMARK_IMAGE", "unknown"),
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("grpc-benchmark-results"))
    parser.add_argument(
        "--flush-cache-url",
        help="HTTP server base URL; flushes the shared runtime before each protocol",
    )
    parser.add_argument("--post-flush-delay", type=float, default=1.0)
    parser.add_argument("--require-conformance", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    raise SystemExit(asyncio.run(_main(parse_args())))


if __name__ == "__main__":
    main()
