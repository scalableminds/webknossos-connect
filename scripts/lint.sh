#!/bin/bash

set -eEuo pipefail

pylint --errors-only --jobs 0 stubs
pylint --errors-only --jobs 0 wkconnect benchmarks build.py
flake8 wkconnect benchmarks stubs build.py
