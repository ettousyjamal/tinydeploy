"""Normalized candidate profiles and importers for profiler and board outputs.

A profile is a small JSON document describing one compressed model on one
target: task score, peak SRAM, latency and where those numbers came from
(``provenance.kind``):

* ``profiler_estimate`` - device-specific profiler output (e.g. Edge Impulse);
* ``measured``          - on-board timing with a documented protocol and raw samples;
* ``synthetic``         - generated fixtures for demonstrations and tests.

The schema is deliberately independent of any vendor export format; the
importers below translate vendor or board outputs into it.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from ._util import TinyDeployError, check_int, check_number, digest, file_digest, read_json

SCHEMA_VERSION = 1
METRICS = ("accuracy", "auc")
PROVENANCE_KINDS = ("measured", "profiler_estimate", "synthetic")
PROTOCOL_KEYS = ("warmup", "repetitions", "timer", "compiler", "flags", "clock_hz")


def validate_profile(profile: dict[str, Any], base: str | Path) -> dict[str, Any]:
    """Check a profile and return it with hashes of its cited source files.

    ``base`` is the directory that relative ``provenance.sources`` paths are
    resolved against. Measured profiles must carry the timing protocol and raw
    samples whose mean agrees with ``latency_ms``; their sample standard
    deviation is added as ``latency_stdev_ms``.
    """
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise TinyDeployError(f"profile schema_version must be {SCHEMA_VERSION}")
    if profile.get("metric") not in METRICS:
        raise TinyDeployError(f"metric must be one of {METRICS}")
    check_number(profile.get("score"), "score", 0, 1)
    check_number(profile.get("peak_sram_bytes"), "peak_sram_bytes")
    check_number(profile.get("latency_ms"), "latency_ms")
    provenance = profile.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in PROVENANCE_KINDS:
        raise TinyDeployError(f"provenance.kind must be one of {PROVENANCE_KINDS}")
    sources = provenance.get("sources", [])
    if provenance["kind"] != "synthetic" and not sources:
        raise TinyDeployError("non-synthetic profiles must cite source files")
    artifacts = []
    for source in sources:
        path = (Path(base) / source).resolve()
        artifacts.append({"path": str(path), "sha256": file_digest(path)})
    if provenance["kind"] == "measured":
        protocol = provenance.get("protocol", {})
        for key in PROTOCOL_KEYS:
            if key not in protocol:
                raise TinyDeployError(f"measured profile missing protocol.{key}")
        check_number(protocol["warmup"], "warmup")
        check_number(protocol["clock_hz"], "clock_hz", 1)
        samples = profile.get("latency_samples_ms", [])
        if len(samples) < 2 or protocol["repetitions"] != len(samples):
            raise TinyDeployError("latency samples must match repetitions and contain >= 2 values")
        for sample in samples:
            check_number(sample, "latency sample")
        mean = statistics.mean(samples)
        if not math.isclose(mean, profile["latency_ms"], rel_tol=0.01, abs_tol=0.001):
            raise TinyDeployError("latency_ms disagrees with the mean of the raw samples")
        profile = dict(profile, latency_stdev_ms=statistics.stdev(samples))
    return dict(profile, source_artifacts=artifacts)


def from_edge_impulse(report_path: str | Path, model_path: str | Path, *, target: str,
                      metric: str, score: float, engine: str = "eon") -> dict[str, Any]:
    """Build a profile from an Edge Impulse device-profiling JSON report.

    ``report_path`` is the JSON returned by Edge Impulse's model profiling for
    one device (``profileInfo.int8`` with ``timePerInferenceMs`` and
    ``memory.{eon,tflite}.{ram,rom}``); the full API response, its ``model``
    object or ``modelInfo.model`` are all accepted. ``engine`` selects the EON
    compiler or TFLM memory figures. The task ``score`` comes from the
    validation split and is passed in explicitly.
    """
    if engine not in ("eon", "tflite"):
        raise TinyDeployError("engine must be 'eon' or 'tflite'")
    if metric not in METRICS:
        raise TinyDeployError(f"metric must be one of {METRICS}")
    report = read_json(report_path)
    model = report.get("modelInfo", report).get("model", report.get("modelInfo", report))
    try:
        info = model["profileInfo"]["int8"]
        latency = info["timePerInferenceMs"]
        memory = info["memory"][engine]
    except (KeyError, TypeError) as exc:
        raise TinyDeployError("not an Edge Impulse INT8 profiling report") from exc
    model_bytes = Path(model_path).read_bytes()
    return {
        "schema_version": SCHEMA_VERSION, "target": target, "metric": metric,
        "score": check_number(score, "score", 0, 1),
        "peak_sram_bytes": check_number(memory["ram"], "ram"),
        "flash_bytes": check_number(memory["rom"], "rom"),
        "latency_ms": check_number(latency, "timePerInferenceMs"),
        "model_sha256": digest(model_bytes),
        "provenance": {"kind": "profiler_estimate", "tool": "edge_impulse", "engine": engine,
                       "device": info.get("device"),
                       "sources": [str(Path(report_path).resolve())]},
    }


def from_benchmark_csv(csv_path: str | Path, model_path: str | Path, *, target: str,
                       metric: str, score: float, peak_sram_bytes: int,
                       protocol: dict[str, Any]) -> dict[str, Any]:
    """Build a measured profile from the CSV emitted by ``firmware/tinydeploy_benchmark.h``.

    The CSV has columns ``sample_index, elapsed_ticks, timer_hz``, one row per
    timed inference. ``protocol`` documents warm-up, timer, compiler, flags and
    clock; its ``repetitions`` must equal the number of rows.
    """
    with Path(csv_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = []
    for expected, row in enumerate(rows):
        if int(row["sample_index"]) != expected:
            raise TinyDeployError("sample_index values must be consecutive from zero")
        ticks = check_number(float(row["elapsed_ticks"]), "elapsed_ticks", 1)
        hz = check_number(float(row["timer_hz"]), "timer_hz", 1)
        samples.append(1000.0 * ticks / hz)
    check_int(protocol.get("repetitions"), "repetitions", 2)
    if protocol["repetitions"] != len(samples):
        raise TinyDeployError("protocol.repetitions differs from the number of CSV rows")
    return {
        "schema_version": SCHEMA_VERSION, "target": target, "metric": metric,
        "score": check_number(score, "score", 0, 1),
        "peak_sram_bytes": check_number(peak_sram_bytes, "peak_sram_bytes"),
        "latency_ms": statistics.mean(samples), "latency_samples_ms": samples,
        "latency_min_ms": min(samples), "latency_max_ms": max(samples),
        "model_sha256": digest(Path(model_path).read_bytes()),
        "provenance": {"kind": "measured", "protocol": protocol,
                       "sources": [str(Path(csv_path).resolve())]},
    }
