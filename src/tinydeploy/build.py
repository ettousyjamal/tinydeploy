"""Target build stage: emit the selected model as a C array and invoke the board build.

Graph optimizations, kernel selection and arena planning belong to the TFLite
converter, TensorFlow Lite Micro and CMSIS-NN inside the board's firmware
project. TinyDeploy only hands the model over, runs the build command and
archives its log and the resulting firmware hash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._util import TinyDeployError, digest, log, run_command


def c_header(model: bytes, symbol: str = "tinydeploy_model") -> str:
    """Return a C/C++ header holding ``model`` as a 16-byte aligned byte array."""
    rows = (", ".join(f"0x{byte:02x}" for byte in model[i:i + 16])
            for i in range(0, len(model), 16))
    body = ",\n".join(rows)
    return ("#pragma once\n#include <stddef.h>\n"
            "#ifdef __cplusplus\nalignas(16)\n#endif\n"
            f"static const unsigned char {symbol}[] = {{\n{body}\n}};\n"
            f"static const size_t {symbol}_size = sizeof({symbol});\n")


def build_firmware(build: dict[str, Any], header: Path, base: Path, output: Path,
                   flash_bytes: int) -> dict[str, Any]:
    """Run the configured build command and copy the firmware into ``output``.

    ``build["command"]`` is an argument list in which ``{model_header}`` is
    replaced by the generated header path; ``build["binary"]`` is the firmware
    file it produces, relative to the configuration directory.
    """
    argv = [part.replace("{model_header}", str(header)) for part in build["command"]]
    elapsed = run_command(argv, base, output / "logs" / "build.json")
    binary = (base / build["binary"]).resolve()
    data = binary.read_bytes()
    if not data or len(data) > flash_bytes:
        raise TinyDeployError("firmware image is empty or larger than target flash")
    (output / "firmware.bin").write_bytes(data)
    log.info("firmware %d B sha256=%s (build %.2f s)", len(data), digest(data)[:12], elapsed)
    return {"bytes": len(data), "sha256": digest(data), "build_elapsed_s": elapsed}
