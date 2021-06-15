from pathlib import Path
from typing import Any, Dict

from setuptools_rust import RustExtension


def build(setup_kwargs: Dict[str, Any]) -> None:
    """
    This `build` function is mandatory in order to build the extensions, since poetry imports it from this module.
    """
    setup_kwargs.update(
        {
            "rust_extensions": [
                RustExtension(
                    "wkconnect.fast_wkw",
                    path=(Path("fast_wkw") / "Cargo.toml"),
                    quiet=False,
                )
            ],
            "zip_safe": False,
        }
    )
