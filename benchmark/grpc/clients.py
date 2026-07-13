"""HTTP and native gRPC streaming clients for the benchmark harness."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
