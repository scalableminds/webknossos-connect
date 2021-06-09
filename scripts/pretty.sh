#!/bin/bash

set -eEuo pipefail

autoflake \
  --in-place --recursive \
  --remove-all-unused-imports --remove-unused-variables \
  wkconnect benchmarks stubs build.py

black wkconnect benchmarks stubs build.py
isort wkconnect benchmarks build.py
