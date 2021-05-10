# -*- coding: utf-8 -*-

from pathlib import Path

try:
    __SPHINX_SETUP__
except NameError:
    __SPHINX_SETUP__ = False

# global namespace
try:
    from sphinx import utils
    from sphinx import augmentation
except ImportError:
    if __SPHINX_SETUP__ is False:
        raise

try:
    version_path = Path(__file__).parent / "VERSION"
    version = version_path.read_text().strip()
except FileNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = (
    "__version__",
    "augmentation",
    "utils",
)

del Path
