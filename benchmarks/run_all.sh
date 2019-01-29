#!/usr/bin/env bash

for path in benchmarks/*.py; do
    file=$(basename $path)
    name="${file%.*}"
    echo -n "$name "
    python3 -m timeit --setup "import benchmarks.$name" "benchmarks.$name.timeit()" 2> /dev/null
done
