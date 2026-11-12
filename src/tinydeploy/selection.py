"""Constraint-based candidate selection (Algorithm 1) and the ``run`` pipeline.

Pipeline: candidate generation -> profiling -> budget filtering -> selection
-> C header and optional target build. Every candidate, including rejected and
skipped ones, is recorded with its reasons in ``results.json``.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

from . import __version__
from ._util import (TinyDeployError, check_id, check_number, digest, log, read_json,
                    run_command, write_json)
from .build import build_firmware, c_header
from .grid import expand_grid
from .profiles import METRICS, validate_profile

TARGET_KEYS = ("sram_bytes", "flash_bytes", "max_latency_ms", "current_ma", "voltage_v")


def energy_uj(latency_ms: float, current_ma: float, voltage_v: float) -> float:
    """Energy per inference E = t x I x V; ms x mA x V gives microjoules."""
    return latency_ms * current_ma * voltage_v


def rejection_reasons(row: dict[str, Any], target: dict[str, Any], minimum_score: float) -> list[str]:
    """Names of the budgets a candidate violates (empty when feasible)."""
    checks = [(row["score"] < minimum_score, "score"),
              (row["peak_sram_bytes"] > target["sram_bytes"], "sram"),
              (row["model_size_bytes"] + target.get("firmware_reserve_bytes", 0)
               > target["flash_bytes"], "flash"),
              (row["latency_ms"] > target["max_latency_ms"], "latency")]
    return [name for failed, name in checks if failed]


def pareto(rows: list[dict[str, Any]]) -> list[str]:
    """Ids of nondominated rows: maximize score; minimize size, SRAM and latency."""
    def objectives(row: dict[str, Any]) -> tuple[float, ...]:
        return (-row["score"], row["model_size_bytes"], row["peak_sram_bytes"], row["latency_ms"])

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        pairs = list(zip(objectives(a), objectives(b)))
        return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)

    return [row["id"] for row in rows if not any(dominates(other, row) for other in rows)]


def select(feasible: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Maximum-score feasible candidate; ties go to lower energy, then smaller size."""
    if not feasible:
        return None
    return min(feasible, key=lambda r: (-r["score"], r["estimated_energy_uj"],
                                        r["model_size_bytes"], r["id"]))


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    target = config["target"]
    check_id(target.get("id"), "target id")
    for key in TARGET_KEYS:
        check_number(target[key], key)
    check_number(target.get("firmware_reserve_bytes", 0), "firmware_reserve_bytes")
    check_number(config["minimum_score"], "minimum_score", 0, 1)
    if config.get("metric") not in METRICS:
        raise TinyDeployError(f"metric must be one of {METRICS}")
    if not config.get("candidates") and "grid" not in config:
        raise TinyDeployError("configuration needs 'candidates', a 'grid' section, or both")
    return target


def _evaluate(spec: dict[str, Any], config: dict[str, Any], target: dict[str, Any],
              base: Path, output: Path, allow_synthetic: bool) -> dict[str, Any]:
    """Run a candidate's commands, then validate and score its profile."""
    cid = spec["id"]
    commands = spec.get("commands") or ([spec["command"]] if "command" in spec else [])
    steps = ("compress", "profile") if len(commands) == 2 else ("worker",) * len(commands)
    if "workdir" in spec:
        Path(spec["workdir"]).mkdir(parents=True, exist_ok=True)
    for index, (step, argv) in enumerate(zip(steps, commands)):
        suffix = step if len(set(steps)) == len(steps) else f"{step}-{index}"
        run_command(argv, base, output / "logs" / f"{cid}-{suffix}.json")
    model_path = (base / spec["model"]).resolve()
    profile_path = (base / spec["profile"]).resolve()
    profile = validate_profile(read_json(profile_path), profile_path.parent)
    if profile["target"] != target["id"] or profile["metric"] != config["metric"]:
        raise TinyDeployError(f"{cid}: profile target or metric does not match the configuration")
    if profile["provenance"]["kind"] == "synthetic" and not allow_synthetic:
        raise TinyDeployError(f"{cid}: synthetic profile requires allow_synthetic / --allow-synthetic")
    model = model_path.read_bytes()
    if not model:
        raise TinyDeployError(f"{cid}: empty model file")
    model_hash = digest(model)
    if profile.get("model_sha256") != model_hash:
        raise TinyDeployError(f"{cid}: model hash mismatch between profile and model file")
    row = {**profile, "id": cid, "model": str(model_path), "model_sha256": model_hash,
           "model_size_bytes": len(model),
           "estimated_energy_uj": energy_uj(profile["latency_ms"], target["current_ma"],
                                            target["voltage_v"])}
    for key in ("bits", "prune_ratio"):
        if key in spec:
            row[key] = spec[key]
    row["rejection_reasons"] = rejection_reasons(row, target, config["minimum_score"])
    write_json(output / "profiles" / f"{cid}.json", profile)
    log.info("candidate %-12s size=%7d B sram=%7d B latency=%8.2f ms energy=%9.1f uJ score=%.4f [%s] -> %s",
             cid, row["model_size_bytes"], row["peak_sram_bytes"], row["latency_ms"],
             row["estimated_energy_uj"], row["score"], profile["provenance"]["kind"],
             "REJECTED (" + ", ".join(row["rejection_reasons"]) + ")" if row["rejection_reasons"] else "feasible")
    return row


def run(config_path: str | Path, output: str | Path, allow_synthetic: bool = False) -> dict[str, Any]:
    """Execute the selection pipeline described by a JSON configuration.

    ``output`` must not exist yet, so earlier evidence is never overwritten.
    Returns the run record that is also written to ``output/results.json``.
    """
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = read_json(config_path)
    target = _validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    log.info("run start: config=%s sha256=%s", config_path.name, digest(config_path.read_bytes())[:12])
    log.info("target %s: sram<=%d B flash<=%d B (reserve %d B) latency<=%g ms min_%s=%g",
             target["id"], target["sram_bytes"], target["flash_bytes"],
             target.get("firmware_reserve_bytes", 0), target["max_latency_ms"],
             config["metric"], config["minimum_score"])

    specs = list(config.get("candidates", []))
    skipped: list[dict[str, Any]] = []
    if "grid" in config:
        grid_specs, skipped = expand_grid(config["grid"], target, base, output)
        specs.extend(grid_specs)
        log.info("grid: %d configurations, %d supported, %d skipped",
                 len(grid_specs) + len(skipped), len(grid_specs), len(skipped))
        for item in skipped:
            log.info("candidate %-12s SKIPPED (%s)", item["id"], item["reason"])
    seen: set[str] = set()
    rows = []
    for spec in specs:
        cid = check_id(spec.get("id"))
        if cid in seen:
            raise TinyDeployError(f"duplicate candidate id: {cid}")
        seen.add(cid)
        rows.append(_evaluate(spec, config, target, base, output, allow_synthetic))

    feasible = [row for row in rows if not row["rejection_reasons"]]
    chosen = select(feasible)
    result: dict[str, Any] = {
        "schema_version": 1, "tinydeploy": __version__, "python": platform.python_version(),
        "config_sha256": digest(config_path.read_bytes()), "target": target,
        "candidates": rows, "skipped": skipped, "feasible_ids": [r["id"] for r in feasible],
        "pareto_ids": pareto(feasible), "selected": chosen["id"] if chosen else None,
        "contains_synthetic": any(r["provenance"]["kind"] == "synthetic" for r in rows),
    }
    log.info("feasible=%s pareto=%s selected=%s", result["feasible_ids"], result["pareto_ids"],
             result["selected"])
    if chosen:
        model = Path(chosen["model"]).read_bytes()
        (output / "selected_model.bin").write_bytes(model)
        header = output / "model_data.h"
        header.write_text(c_header(model))
        log.info("emitted %s (%d B model, sha256=%s)", header.name, len(model), digest(model)[:12])
        if "build" in config:
            result["firmware"] = build_firmware(config["build"], header, base, output,
                                                target["flash_bytes"])
    result["wall_time_s"] = time.perf_counter() - started
    write_json(output / "results.json", result)
    log.info("run done in %.3f s -> %s", result["wall_time_s"], output / "results.json")
    return result
