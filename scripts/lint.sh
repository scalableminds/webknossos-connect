#!/bin/bash

set -eEuo pipefail

pylint --errors-only --jobs 0 stubs
pylint --errors-only --jobs 0 wkconnect benchmarks
flake8 wkconnect benchmarks stubs
