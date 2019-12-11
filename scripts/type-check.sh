#!/bin/bash

set -eEuo pipefail

mypy \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  wkconnect benchmarks stubs
