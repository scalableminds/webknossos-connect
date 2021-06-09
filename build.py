from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import os
import subprocess
import shutil
import platform
from pathlib import Path
from typing import Any, Dict


class BuildPyCommand(build_py):
    """Modified build command to compile `fast_wkw` Rust library."""

    def __build_rust_library(self):
        this_dir = Path(__file__).parent
        rust_dir = this_dir / "fast_wkw"

        # building Rust library
        subprocess.call(["cargo", "clean"], cwd=rust_dir)
        subprocess.call(["cargo", "build", "--release"], cwd=rust_dir)

        lib_name_platform = {
            "Linux": ("libfast_wkw.so", "fast_wkw.so"),
            "Windows": ("libfast_wkw.dll", "fast_wkw.pyd"),
            "Darwin": ("libfast_wkw.dylib", "fast_wkw.so"),
        }
        lib_name = lib_name_platform[platform.system()]
        lib_file = rust_dir / "target" / "release" / lib_name[0]

        # copying to lib dir
        shutil.copy(lib_file, this_dir / "wkconnect" / lib_name[1])

    def run(self):
        self.__build_rust_library()
        build_py.run(self)


def build(setup_kwargs: Dict[str, Any]) -> None:
    """
    This `build` function is mandatory in order to build the extensions, since poetry imports it from this module.
    """
    setup_kwargs.update({"cmdclass": {"build_py": BuildPyCommand}})
