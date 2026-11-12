"""Shared helpers: hashing, JSON I/O, input validation and logged subprocesses."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("tinydeploy")

ID_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class TinyDeployError(ValueError):
    """Raised for invalid inputs, failed commands and integrity violations."""


def digest(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Any) -> None:
    """Write indented JSON; NaN and infinity are rejected to keep records portable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def check_number(value: Any, name: str, minimum: float = 0, maximum: float | None = None) -> float:
    """Validate a finite real number within ``[minimum, maximum]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinyDeployError(f"{name} must be a number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise TinyDeployError(f"invalid {name}: {value}")
    return value


def check_int(value: Any, name: str, minimum: int = 0) -> int:
    """Validate an integer (not a bool) that is at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TinyDeployError(f"{name} must be an integer >= {minimum}")
    return value


def check_id(value: Any, name: str = "candidate id") -> str:
    """Identifiers become file names, so only letters, digits, '_' and '-' are allowed."""
    if not isinstance(value, str) or not value or any(c not in ID_CHARACTERS for c in value):
        raise TinyDeployError(f"{name} must contain only letters, digits, '_' or '-'")
    return value


def run_command(argv: Sequence[str], cwd: str | Path, log_path: str | Path, timeout: float = 600) -> float:
    """Run an explicit argument vector (never a shell) and archive a JSON log.

    The log records argv, working directory, return code, elapsed time, stdout
    and stderr, and is written for failures as well as successes. Returns the
    elapsed wall time in seconds.
    """
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(x, str) for x in argv):
        raise TinyDeployError("command must be a nonempty list of strings")
    argv = list(argv)
    start = time.perf_counter()
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        write_json(log_path, {"argv": argv, "cwd": str(cwd), "status": "timeout", "timeout_s": timeout})
        log.error("command timed out after %.0f s: %s", timeout, " ".join(argv))
        raise TinyDeployError(f"command timed out; see {log_path}") from exc
    elapsed = time.perf_counter() - start
    write_json(log_path, {"argv": argv, "cwd": str(cwd), "returncode": result.returncode,
                          "elapsed_s": elapsed, "stdout": result.stdout, "stderr": result.stderr})
    shown = " ".join(Path(a).name if "/" in a or "\\" in a else a for a in argv)
    log.info("exec rc=%d %.3fs %s", result.returncode, elapsed, shown)
    for line in result.stdout.strip().splitlines()[-3:]:
        log.debug("  stdout: %s", line)
    if result.returncode:
        log.error("command failed (exit %d); log: %s", result.returncode, log_path)
        raise TinyDeployError(f"command failed with exit code {result.returncode}; see {log_path}")
    return elapsed
