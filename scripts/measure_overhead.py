"""Measure TinyDeploy's own host-side overhead on the example inputs. Standard library only.

Measures, on the example inputs (training, profiling and compilation excluded):
  * the selection pipeline called in-process (10 runs);
  * one ``tinydeploy run`` command, including interpreter start-up (10 runs);
  * the 3,000-event OTA fleet simulation (5 runs);
  * peak resident memory of the OTA simulation process (Linux/macOS).

Usage:  python scripts/measure_overhead.py OUTPUT_DIR
Writes OUTPUT_DIR/overhead.json and prints a summary. Times depend on the machine.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tinydeploy import __version__, run, simulate
from tinydeploy._util import read_json, write_json
from tinydeploy.demo import prepare_ota, prepare_selection


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "stdev": statistics.stdev(values),
            "min": min(values), "max": max(values), "runs": len(values)}


def main(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = prepare_selection(output / "inputs" / "selection")
    old, new, ota_config = prepare_ota(output / "inputs" / "ota")

    in_process_ms, cli_ms, ota_s = [], [], []
    for i in range(10):
        start = time.perf_counter()
        run(config, output / "runs" / f"api-{i}", allow_synthetic=True)
        in_process_ms.append(1000 * (time.perf_counter() - start))
        start = time.perf_counter()
        subprocess.run([sys.executable, "-m", "tinydeploy", "run", str(config), "--output",
                        str(output / "runs" / f"cli-{i}"), "--allow-synthetic"],
                       check=True, capture_output=True)
        cli_ms.append(1000 * (time.perf_counter() - start))
    for i in range(5):
        start = time.perf_counter()
        simulate(old, new, read_json(ota_config), output / "runs" / f"ota-{i}")
        ota_s.append(time.perf_counter() - start)

    peak_mb = None
    try:
        import resource
        subprocess.run([sys.executable, "-m", "tinydeploy", "simulate-ota", str(old), str(new),
                        "--config", str(ota_config), "--output", str(output / "runs" / "ota-memory")],
                       check=True, capture_output=True)
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_mb = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except ImportError:  # Windows
        pass

    selection_logic_ms = [read_json(output / "runs" / f"api-{i}" / "results.json")["wall_time_s"] * 1000
                          for i in range(10)]
    result = {
        "tinydeploy": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
        "selection_logic_ms": stats(selection_logic_ms),
        "selection_in_process_ms": stats(in_process_ms),
        "cli_run_ms": stats(cli_ms),
        "ota_simulation_s": stats(ota_s),
        "ota_peak_rss_mb": peak_mb,
    }
    write_json(output / "overhead.json", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    r = main(Path(sys.argv[1]))
    print(f"selection logic        {r['selection_logic_ms']['mean']:.1f} ms")
    print(f"tinydeploy run (CLI)   {r['cli_run_ms']['mean']:.1f} +/- {r['cli_run_ms']['stdev']:.1f} ms")
    print(f"OTA simulation         {r['ota_simulation_s']['mean']:.2f} +/- {r['ota_simulation_s']['stdev']:.2f} s")
    if r["ota_peak_rss_mb"] is not None:
        print(f"OTA peak memory        {r['ota_peak_rss_mb']:.0f} MB")
