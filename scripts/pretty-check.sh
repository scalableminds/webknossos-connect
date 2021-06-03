#!/bin/bash

set -eEuo pipefail

black --check wkconnect benchmarks stubs
isort --check-only wkconnect benchmarks
