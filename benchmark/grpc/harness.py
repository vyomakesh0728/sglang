#!/usr/bin/env python3
"""Benchmark HTTP /generate and native TextGenerate with one workload.

The harness intentionally lives outside the production client package. Native Python
stubs are generated from the repository's canonical proto into a temporary directory.
This keeps generated code out of the tree and makes proto drift visible immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.grpc.clients import (  # noqa: E402
    SCHEMA_VERSION,
    Client,
    HttpGenerateClient,
    NativeGrpcClient,
    RequestResult,
    RequestSpec,
)
from benchmark.grpc.conformance import (  # noqa: E402
    classify_repeatability,
    compare_results,
)
from benchmark.grpc.metrics import _percentiles, summarize  # noqa: E402


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
