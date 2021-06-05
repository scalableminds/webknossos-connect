#!/bin/bash

set -eEuo pipefail

autoflake \
  --in-place --recursive \
  --remove-all-unused-imports --remove-unused-variables \
  wkconnect benchmarks stubs

black wkconnect benchmarks stubs
isort wkconnect benchmarks
