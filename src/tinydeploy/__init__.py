"""TinyDeploy: constraint-based TinyML deployment orchestration.

Pipeline stages and their modules:

* candidate generation - :mod:`tinydeploy.grid` (Algorithm 1 grid),
  :mod:`tinydeploy.compression` (optional TensorFlow adapter);
* profiling            - :mod:`tinydeploy.profiles` (normalized profiles, importers);
* filtering/selection  - :mod:`tinydeploy.selection`;
* target build         - :mod:`tinydeploy.build`;
* OTA preparation      - :mod:`tinydeploy.delta`, :mod:`tinydeploy.ota`.

The core uses only the Python standard library.
"""

__version__ = "1.0.0"

from ._util import TinyDeployError  # noqa: E402
from .delta import apply_delta, make_delta  # noqa: E402
from .ota import simulate  # noqa: E402
from .selection import run  # noqa: E402

__all__ = ["TinyDeployError", "apply_delta", "make_delta", "run", "simulate", "__version__"]
