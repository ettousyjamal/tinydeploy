"""Hash-bound sparse delta codec for firmware images.

A delta lists the byte runs that differ between an old and a new image. It is
bound to both images by SHA-256: applying it to any other base, or producing an
image whose hash differs from the declared one, is rejected.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ._util import TinyDeployError, digest

DELTA_SCHEMA_VERSION = 1


def make_delta(old: bytes, new: bytes) -> dict[str, Any]:
    """Encode the changed byte runs of ``new`` relative to ``old``."""
    runs = []
    index = 0
    while index < len(new):
        if index < len(old) and new[index] == old[index]:
            index += 1
            continue
        start = index
        while index < len(new) and (index >= len(old) or new[index] != old[index]):
            index += 1
        runs.append([start, base64.b64encode(new[start:index]).decode("ascii")])
    return {"schema_version": DELTA_SCHEMA_VERSION, "base_sha256": digest(old),
            "new_sha256": digest(new), "new_size": len(new), "runs": runs}


def wire(delta: dict[str, Any]) -> bytes:
    """Compact JSON encoding used as the transmitted payload (includes base64 overhead)."""
    return json.dumps(delta, separators=(",", ":"), allow_nan=False).encode()


def apply_delta(old: bytes, delta: dict[str, Any]) -> bytes:
    """Reconstruct the new image, verifying the base and result hashes."""
    if delta.get("schema_version") != DELTA_SCHEMA_VERSION or digest(old) != delta.get("base_sha256"):
        raise TinyDeployError("delta schema version or base image hash mismatch")
    size = delta["new_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TinyDeployError("invalid reconstructed image size")
    output = bytearray(old[:size])
    output.extend(b"\0" * (size - len(output)))
    previous_end = 0
    for offset, encoded in delta["runs"]:
        data = base64.b64decode(encoded, validate=True)
        if (isinstance(offset, bool) or not isinstance(offset, int) or offset < previous_end
                or offset + len(data) > size):
            raise TinyDeployError("overlapping or out-of-range delta run")
        output[offset:offset + len(data)] = data
        previous_end = offset + len(data)
    if digest(bytes(output)) != delta["new_sha256"]:
        raise TinyDeployError("reconstructed image hash mismatch")
    return bytes(output)
