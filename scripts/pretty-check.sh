#!/bin/bash

set -eEuo pipefail

black --check wkconnect benchmarks stubs build.py
isort --check-only wkconnect benchmarks build.py
