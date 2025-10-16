#!/bin/bash

# Hyperdrive ANN Benchmark Script

set -e

BIN_DIR="../build"
GEN_DATA="$BIN_DIR/gen_data"
HYPERDRIVE_ANN="$BIN_DIR/hyperdrive-ann"

echo "=== Hyperdrive ANN Benchmark Suite ==="
echo

# Check if binaries exist
if [ ! -f "$GEN_DATA" ]; then
    echo "ERROR: gen_data binary not found at $GEN_DATA"
    echo "Please build the project first: mkdir build && cd build && cmake .. && make"
    exit 1
fi

if [ ! -f "$HYPERDRIVE_ANN" ]; then
    echo "ERROR: hyperdrive-ann binary not found at $HYPERDRIVE_ANN"
    echo "Please build the project first: mkdir build && cd build && cmake .. && make"
    exit 1
fi

# Create data directory
mkdir -p data

echo "Generating test datasets..."

# Generate database (1M vectors x 128 dim)
echo "  - Database: 1,000,000 vectors x 128 dimensions"
$GEN_DATA generate 1000000 128 data/database.bin

# Generate queries (16 vectors x 128 dim)
echo "  - Queries: 16 vectors x 128 dimensions"
$GEN_DATA generate 16 128 data/queries.bin

echo

# Run benchmarks with different configurations
echo "Running benchmarks..."

echo "1. Default configuration (tile size 256):"
$HYPERDRIVE_ANN --benchmark data/database.bin data/queries.bin 10 --iterations 3 --warmup 1

echo
echo "2. Optimized tile size (512):"
$HYPERDRIVE_ANN --benchmark data/database.bin data/queries.bin 10 --tile-size 512 --iterations 3 --warmup 1

echo
echo "3. Small tile size (128):"
$HYPERDRIVE_ANN --benchmark data/database.bin data/queries.bin 10 --tile-size 128 --iterations 3 --warmup 1

echo
echo "4. Large k value (50):"
$HYPERDRIVE_ANN --benchmark data/database.bin data/queries.bin 50 --iterations 3 --warmup 1

echo
echo "5. Performance test (10 iterations):"
$HYPERDRIVE_ANN --benchmark data/database.bin data/queries.bin 10 --iterations 10 --warmup 2

echo
echo "=== Benchmark Complete ==="
echo "Results saved to data/ directory"
echo "Clean up with: rm -rf data/"


