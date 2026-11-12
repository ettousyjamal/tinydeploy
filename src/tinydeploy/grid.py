"""Compression grid of Algorithm 1: bit-width x pruning ratio, with support checks.

The paper's grid is bit-width {8, 4} x pruning ratio {0, 0.3, 0.5, 0.7}, i.e.
eight configurations per target *before* support checks. Each supported
configuration becomes one candidate whose model and profile are produced by
external commands (``compress`` and ``profile``); unsupported configurations
are recorded as skipped instead of silently disappearing.

Command templates may use these placeholders:

``{base_model}``  model to compress (from the grid section);
``{bits}``        quantization bit-width, e.g. ``8``;
``{prune}``       pruning ratio as a decimal, e.g. ``0.3``;
``{prune_pct}``   pruning ratio in percent, e.g. ``30``;
``{workdir}``     per-candidate directory inside the run output;
``{model}``       expected model file (``model_file`` template, resolved);
``{profile}``     expected profile file (``profile_file`` template, resolved);
``{target}``      target identifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._util import TinyDeployError, check_int, check_number

PAPER_BIT_WIDTHS = (8, 4)
PAPER_PRUNE_RATIOS = (0.0, 0.3, 0.5, 0.7)


def candidate_id(bits: int, prune: float) -> str:
    """Stable identifier, e.g. ``int8-p30`` for INT8 with 30% pruning."""
    return f"int{bits}-p{round(prune * 100):02d}"


def _fill(template: Any, values: dict[str, str]) -> Any:
    if isinstance(template, str):
        try:
            return template.format(**values)
        except (KeyError, IndexError) as exc:
            raise TinyDeployError(f"unknown placeholder in command template: {template}") from exc
    if isinstance(template, list):
        return [_fill(item, values) for item in template]
    raise TinyDeployError("command templates must be strings or lists of strings")


def expand_grid(grid: dict[str, Any], target: dict[str, Any], base: Path,
                output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand a grid section into (candidates, skipped configurations).

    ``target["supported_bit_widths"]`` lists the bit-widths the target's
    runtime can execute (default: ``[8]``, since TFLM/CMSIS-NN on Cortex-M4 run
    INT8 kernels). Unsupported bit-widths are reported as skipped.
    """
    bit_widths = grid.get("bit_widths", list(PAPER_BIT_WIDTHS))
    ratios = grid.get("prune_ratios", list(PAPER_PRUNE_RATIOS))
    supported = set(target.get("supported_bit_widths", [8]))
    for key in ("compress", "profile", "model_file", "profile_file"):
        if key not in grid:
            raise TinyDeployError(f"grid section is missing '{key}'")
    candidates, skipped = [], []
    for bits in bit_widths:
        check_int(bits, "bit width", 1)
        for prune in ratios:
            check_number(prune, "prune ratio", 0, 0.99)
            cid = candidate_id(bits, prune)
            if bits not in supported:
                skipped.append({"id": cid, "bits": bits, "prune_ratio": prune,
                                "reason": f"INT{bits} not supported on target {target['id']}"})
                continue
            workdir = output / "candidates" / cid
            values = {"base_model": str((base / grid.get("base_model", "")).resolve()),
                      "bits": str(bits), "prune": f"{prune:g}",
                      "prune_pct": str(round(prune * 100)), "workdir": str(workdir),
                      "target": str(target["id"])}
            values["model"] = _fill(grid["model_file"], values)
            values["profile"] = _fill(grid["profile_file"], values)
            candidates.append({"id": cid, "bits": bits, "prune_ratio": prune,
                               "workdir": workdir, "model": values["model"],
                               "profile": values["profile"],
                               "commands": [_fill(grid["compress"], values),
                                            _fill(grid["profile"], values)]})
    return candidates, skipped
