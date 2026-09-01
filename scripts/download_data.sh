#!/usr/bin/env bash
# Downloads the real MNIST and California Housing datasets into data/.
# Linux / macOS / Git Bash. Requires curl and gzip (both standard).
#
# The project runs fine without this script (it falls back to a
# deterministic synthetic dataset), but running it once gives you the
# real data.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mnist_dir="$root/data/mnist"
housing_dir="$root/data/housing"
mkdir -p "$mnist_dir" "$housing_dir"

mnist_base="https://storage.googleapis.com/cvdf-datasets/mnist"
for name in train-images-idx3-ubyte train-labels-idx1-ubyte \
            t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "$mnist_dir/$name" ]; then
        echo "already present: $name"
        continue
    fi
    echo "downloading $name ..."
    curl -sf --max-time 120 -o "$mnist_dir/$name.gz" "$mnist_base/$name.gz"
    gzip -d "$mnist_dir/$name.gz"
done

if [ -f "$housing_dir/housing.csv" ]; then
    echo "already present: housing.csv"
else
    echo "downloading housing.csv ..."
    curl -sf --max-time 120 -o "$housing_dir/housing.csv" \
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
fi

echo
echo "Datasets ready in data/."
