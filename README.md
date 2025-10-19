# hyperdrive-ann

GPU-accelerated approximate nearest neighbor search engine using CUDA 12.8 and C++20.

## Features

### Core Engine
- GPU-accelerated cosine similarity computation using CUDA kernels
- Shared memory tiling for optimal memory bandwidth utilization
- Warp-level reductions for efficient parallel dot product computation
- Occupancy-aware block sizing for maximum GPU utilization
- Batch processing of multiple queries
- CPU baseline for performance comparison

### Production Infrastructure
- Memory pool management for efficient GPU memory allocation
- Comprehensive logging system with multiple levels
- YAML-based configuration management
- Vector quantization support (INT8/INT4)
- Docker containerization with NVIDIA GPU support
- Interactive CLI with performance rating system

## Status

ACTIVE DEVELOPMENT - This project is currently under active development.

NO CONTRIBUTIONS ACCEPTED - Please do not submit pull requests or issues at this time. This is a personal research project.

## Prerequisites

### Required
- CUDA 12.8 or later
- CUB library (CUDA C++ Unbound)
- CMake 3.20 or later
- C++20 compatible compiler

### Optional (for production features)
- Docker with NVIDIA GPU support
- YAML-CPP library
- pybind11 (for Python bindings)

## Quick Start

```bash
# Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Verification

**Super simple - just run:**

```bash
./hyperdrive-ann --go
```

**Interactive flow:**
1. Shows what will be tested (vectors, dimensions, GPU vs CPU)
2. Asks "Do you want to proceed? (Y/n)"
3. Choose test size: Quick (100K vectors) or Full (1M vectors)
4. Confirms again: "Do you want to proceed? (Y/n)"
5. Runs benchmark and shows results

**Or run individual tests:**

```bash
# Run tests (should show "ALL TESTS PASSED")
./test_basic

# Run quick benchmark (should show GPU speedup)
./hyperdrive-ann --benchmark --iterations 1
```

**Expected output:**
- Clean ASCII banner with device info
- Interactive prompts for test configuration
- Performance rating (POOR/FAIR/GOOD/VERY GOOD/EXCELLENT)
- GPU speedup vs CPU comparison
- Correctness verification with [PASS]/[FAIL] indicators

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## API

```cpp
#include"ann.hpp"

// Simple usage
std::vector<std::vector<int>> results=topK(db,dbCount,dim,queries,queryCount,k);

// Advanced usage
HyperdriveANN ann;
ann.initialize(dbCount,dim,queryCount,k,tileSize);
auto results=ann.topK(db,dbCount,dim,queries,queryCount,k);
```

## Docker Support

```bash
# Build and run with Docker
docker-compose up --build

# Run specific services
docker-compose run hyperdrive-benchmark
docker-compose run hyperdrive-test
```

## Performance

### RTX 4060 Laptop GPU Results
- **Database**: 1M vectors × 128 dimensions
- **Queries**: 16 vectors
- **GPU Time**: ~7.7 ms per query
- **CPU Time**: ~479 ms per query
- **Speedup**: **62x faster**
- **Rating**: GOOD performance

## Troubleshooting

**Build fails:** Check CUDA installation and CUB library
**GPU not found:** Verify with `nvidia-smi`
**Poor performance:** Try `--tile-size 512`
**Memory issues:** Enable quantization in config

## License

MIT License - see LICENSE file.
