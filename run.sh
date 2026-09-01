#!/usr/bin/env bash
# One-click entry point (Linux / macOS / Git Bash): fetch the datasets if
# they are not there yet, then build and run the experiment suite.
#
# If the download fails (no internet, firewall), the program still runs on
# a deterministic synthetic fallback and prints that it is doing so.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f data/mnist/train-images-idx3-ubyte ]; then
    echo "Fetching datasets (requires internet, one time only)..."
    if ! bash scripts/download_data.sh; then
        echo
        echo "Download failed - continuing with the synthetic fallback."
        echo
    fi
fi

cargo run --release
