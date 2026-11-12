"""Command-line interface: ``tinydeploy <command> ...`` (or ``python -m tinydeploy``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from ._util import TinyDeployError, digest, read_json, write_json

EXIT_OK, EXIT_ERROR, EXIT_NO_FEASIBLE = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinydeploy",
        description="Constraint-based TinyML model selection, target build hand-off and OTA simulation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="print execution logs to stderr (-v: info, -vv: debug)")
    parser.add_argument("--log-file", help="also append timestamped execution logs to this file")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("run", help="profile, filter and select candidates; emit C header; optional build")
    p.add_argument("config", help="JSON experiment configuration")
    p.add_argument("--output", required=True, help="new directory for the run record")
    p.add_argument("--allow-synthetic", action="store_true", help="accept profiles marked synthetic")

    p = sub.add_parser("simulate-ota", help="seeded A/B fleet simulation, full image vs delta")
    p.add_argument("old", help="current firmware image")
    p.add_argument("new", help="new firmware image")
    p.add_argument("--config", required=True, help="JSON fleet configuration")
    p.add_argument("--output", required=True, help="new directory for events and summary")

    p = sub.add_parser("delta", help="create a hash-bound delta between two images")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--output", required=True, help="delta JSON file")

    p = sub.add_parser("apply-delta", help="verify the base, rebuild and verify the new image")
    p.add_argument("old")
    p.add_argument("delta")
    p.add_argument("--output", required=True, help="reconstructed image")

    p = sub.add_parser("demo", help="run the self-contained example (generated inputs, all stages)")
    p.add_argument("output", help="new directory for all demo inputs and outputs")

    p = sub.add_parser("import-profile", help="convert profiler or board output into a profile")
    p.add_argument("source", choices=("edge-impulse", "benchmark"),
                   help="edge-impulse: profiling JSON report; benchmark: CSV from tinydeploy_benchmark.h")
    p.add_argument("report", help="Edge Impulse JSON report or benchmark CSV")
    p.add_argument("--model", required=True, help="the profiled model file (hash-bound)")
    p.add_argument("--target", required=True)
    p.add_argument("--metric", choices=("accuracy", "auc"), required=True)
    p.add_argument("--score", type=float, required=True, help="validation score in [0, 1]")
    p.add_argument("--engine", choices=("eon", "tflite"), default="eon", help="edge-impulse only")
    p.add_argument("--peak-sram", type=int, help="benchmark only: measured peak SRAM bytes")
    p.add_argument("--protocol", help="benchmark only: JSON file with the timing protocol")
    p.add_argument("--output", required=True, help="profile JSON file")

    p = sub.add_parser("compress", help="INT8 PTQ with optional pruning/distillation (needs TensorFlow)")
    p.add_argument("model", help="single-input Keras model (.keras)")
    p.add_argument("--data", required=True,
                   help=".npz with x_calibration (and x_train, y_train for --teacher or --fine-tune-epochs)")
    p.add_argument("--output", required=True, help="output .tflite file")
    p.add_argument("--bits", type=int, default=8, help="quantization bit-width (only 8 is supported)")
    p.add_argument("--prune", type=float, default=0.0, help="structured pruning ratio in [0, 1)")
    p.add_argument("--teacher", help="teacher model for knowledge distillation")
    p.add_argument("--epochs", type=int, default=10, help="distillation epochs")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the hard-label loss")
    p.add_argument("--task", choices=("classification", "reconstruction"), default="classification")
    p.add_argument("--seed", type=int, default=0)
    return parser


def _compress(args: argparse.Namespace) -> None:
    from .compression import distill, int8_ptq, prune_dense
    import numpy as np
    import tensorflow as tf

    if args.bits != 8:
        raise TinyDeployError(f"INT{args.bits} is not supported; TFLM/CMSIS-NN targets run INT8")
    started = time.perf_counter()
    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.models.load_model(args.model, compile=False)
    history = []
    with np.load(args.data, allow_pickle=False) as data:
        if args.prune:
            model = prune_dense(model, args.prune)
        if args.teacher:
            teacher = tf.keras.models.load_model(args.teacher, compile=False)
            history = distill(teacher, model, data["x_train"], data["y_train"], epochs=args.epochs,
                              temperature=args.temperature, alpha=args.alpha, seed=args.seed,
                              task=args.task)
        converted = int8_ptq(model, data["x_calibration"], args.output)
    output = Path(args.output)
    write_json(output.with_suffix(".json"), {
        "evidence_kind": "host_conversion", "settings": vars(args),
        "tensorflow": tf.__version__, "input_model_sha256": digest(Path(args.model).read_bytes()),
        "data_sha256": digest(Path(args.data).read_bytes()),
        "teacher_sha256": digest(Path(args.teacher).read_bytes()) if args.teacher else None,
        "output_sha256": digest(converted), "output_bytes": len(converted),
        "elapsed_s": time.perf_counter() - started, "distillation_history": history})
    print(f"wrote {output} ({len(converted)} bytes)")


def _import_profile(args: argparse.Namespace) -> None:
    from .profiles import from_benchmark_csv, from_edge_impulse

    if args.source == "edge-impulse":
        profile = from_edge_impulse(args.report, args.model, target=args.target,
                                    metric=args.metric, score=args.score, engine=args.engine)
    else:
        if args.peak_sram is None or args.protocol is None:
            raise TinyDeployError("benchmark import needs --peak-sram and --protocol")
        profile = from_benchmark_csv(args.report, args.model, target=args.target,
                                     metric=args.metric, score=args.score,
                                     peak_sram_bytes=args.peak_sram,
                                     protocol=read_json(args.protocol))
    write_json(args.output, profile)
    print(f"wrote {args.output} ({profile['provenance']['kind']}, {profile['latency_ms']:.3f} ms)")


LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"


def configure_logging(verbosity: int = 0, log_file: str | Path | None = None) -> None:
    """Send ``tinydeploy`` logs to stderr (by verbosity) and optionally to a file (debug level)."""
    logger = logging.getLogger("tinydeploy")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE)
    console = logging.StreamHandler()
    console.setLevel({0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG))
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    try:
        if args.command == "run":
            from .selection import run
            result = run(args.config, args.output, args.allow_synthetic)
            print(f"candidates={len(result['candidates'])} skipped={len(result['skipped'])} "
                  f"feasible={len(result['feasible_ids'])} selected={result['selected']}")
            return EXIT_OK if result["selected"] else EXIT_NO_FEASIBLE
        if args.command == "simulate-ota":
            from .ota import simulate
            result = simulate(args.old, args.new, read_json(args.config), args.output)
            for mode, totals in result["modes"].items():
                print(f"{mode:>5}: committed={totals['committed']} rolled_back={totals['rolled_back']} "
                      f"on_air={totals['transmitted_bytes']} B")
            print(f"delta energy saving: {result['delta_energy_saving_fraction']:.1%}")
        elif args.command == "delta":
            from .delta import make_delta
            write_json(args.output, make_delta(Path(args.old).read_bytes(), Path(args.new).read_bytes()))
        elif args.command == "apply-delta":
            from .delta import apply_delta
            Path(args.output).write_bytes(apply_delta(Path(args.old).read_bytes(), read_json(args.delta)))
        elif args.command == "demo":
            from .demo import run_demo
            run_demo(args.output)
            print(json.dumps(read_json(Path(args.output) / "demo_summary.json")["grid"], indent=1))
            print(f"demo outputs written to {args.output}")
        elif args.command == "import-profile":
            _import_profile(args)
        elif args.command == "compress":
            _compress(args)
    except (TinyDeployError, KeyError, TypeError, OSError) as exc:
        print(f"tinydeploy: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
