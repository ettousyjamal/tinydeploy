"""Stand-in compressor and profiler used by the grid demonstration.

Real deployments plug Edge Impulse, the TensorFlow adapter or another tool
into the grid's ``compress`` and ``profile`` commands. This adapter replaces
both with deterministic arithmetic so the full Algorithm 1 flow runs anywhere
in seconds. Its models are random bytes and its profiles are labelled
``synthetic``; none of its numbers are benchmark results.

Usage (as called by the demo grid configuration)::

    python -m tinydeploy._demo_adapter compress BASE --bits 8 --prune 0.3 --output model.bin
    python -m tinydeploy._demo_adapter profile model.bin --bits 8 --prune 0.3 --target T --output profile.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ._util import digest, write_json

PARAMETERS = 65_536  # parameter count of the synthetic FP32 base model


def compressed_size(bits: int, prune: float) -> int:
    return int(PARAMETERS * (1 - prune) * bits / 8)


def synthetic_profile(bits: int, prune: float) -> dict[str, float]:
    """Monotone toy trade-offs: pruning saves memory and time but costs score."""
    return {"score": round(0.93 - 0.12 * prune ** 2 - (0.02 if bits < 8 else 0.0), 4),
            "peak_sram_bytes": int(40_000 + 120_000 * (1 - prune)),
            "latency_ms": round(120 * (1 - prune) * bits / 8, 2)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tinydeploy._demo_adapter")
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("compress", "profile"):
        step = sub.add_parser(name)
        step.add_argument("input")
        step.add_argument("--bits", type=int, required=True)
        step.add_argument("--prune", type=float, required=True)
        step.add_argument("--output", required=True)
        if name == "profile":
            step.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.step == "compress":
        seed = args.bits * 1000 + round(args.prune * 100)
        output.write_bytes(random.Random(seed).randbytes(compressed_size(args.bits, args.prune)))
        print(f"compressed INT{args.bits}, prune {args.prune:g}: {output.stat().st_size} bytes")
        return
    model = Path(args.input).read_bytes()
    write_json(output, {"schema_version": 1, "target": args.target, "metric": "accuracy",
                        **synthetic_profile(args.bits, args.prune), "model_sha256": digest(model),
                        "provenance": {"kind": "synthetic", "sources": [],
                                       "note": "Demo adapter output; not a benchmark result"}})
    print(f"profiled {Path(args.input).name}")


if __name__ == "__main__":
    main()
