# Hyperdrive-ANN Architecture

## System Overview

Hyperdrive-ANN is a GPU-accelerated approximate nearest neighbor (ANN) search engine built with CUDA 12.8 and C++20. The system efficiently computes top-k cosine similarity searches across large vector databases using shared-memory tiling, warp-level reductions, and occupancy-aware block sizing.

## High-Level Architecture

```mermaid
graph TB
    CLI[CLI Tool] --> API[C++ API]
    API --> ANN[HyperdriveANN Class]
    ANN --> GPU[GPU Kernels]
    GPU --> CUDA[CUDA Runtime]
    
    subgraph "Data Flow"
        DB[(Database Vectors)]
        Q[(Query Vectors)]
        DB --> ANN
        Q --> ANN
        ANN --> RES[(Top-K Results)]
    end
    
    subgraph "GPU Processing"
        GPU --> SIM[Similarity Kernel]
        GPU --> TOP[Top-K Kernel]
        SIM --> TOP
    end
```

The system follows a layered architecture where the CLI interface provides user interaction, the C++ API offers programmatic access, and the GPU kernels handle the computationally intensive similarity computations. Data flows from input vectors through the ANN engine to produce ranked similarity results.

## CUDA Kernel Architecture

```mermaid
graph LR
    subgraph "Compute Similarities Kernel"
        QV[Query Vector] --> SM[Shared Memory]
        DBV[Database Vectors] --> DOT[Dot Product]
        SM --> DOT
        DOT --> SIM[Similarity Scores]
    end
    
    subgraph "Top-K Selection Kernel"
        SIM --> SORT[Shared Memory Sort]
        SORT --> TK[Top-K Indices]
    end
    
    subgraph "Memory Hierarchy"
        GM[Global Memory] --> SM
        SM --> RM[Registers]
        RM --> WARP[Warp Primitives]
    end
```

The CUDA implementation uses a two-phase approach: first computing cosine similarities between query and database vectors using shared memory for efficient data access, then performing top-k selection using shared memory sorting. The memory hierarchy optimization ensures minimal global memory access while maximizing shared memory utilization for faster computation.

## Performance Optimization Strategy

```mermaid
graph TD
    subgraph "Memory Optimization"
        TILE[Tiled Access] --> COAL[Coalesced Memory]
        COAL --> SM[Shared Memory Cache]
    end
    
    subgraph "Compute Optimization"
        WARP[Warp-Level Reductions] --> BLOCK[Block-Level Operations]
        BLOCK --> OCC[Occupancy-Aware Sizing]
    end
    
    subgraph "Algorithm Optimization"
        NORM[Normalized Vectors] --> COS[Cosine Similarity]
        COS --> TOPK[Efficient Top-K Selection]
    end
    
    TILE --> WARP
    WARP --> NORM
```

The optimization strategy focuses on three key areas: memory access patterns through tiled and coalesced memory operations, compute efficiency via warp-level reductions and occupancy-aware block sizing, and algorithmic efficiency through normalized vector processing and optimized top-k selection algorithms.

## Data Flow and Memory Management

```mermaid
sequenceDiagram
    participant H as Host (CPU)
    participant D as Device (GPU)
    participant K1 as Similarity Kernel
    participant K2 as Top-K Kernel
    
    H->>D: Copy database vectors
    H->>D: Copy query vectors
    H->>K1: Launch similarity computation
    K1->>K1: Load query to shared memory
    K1->>K1: Compute dot products
    K1->>D: Store similarity scores
    H->>K2: Launch top-k selection
    K2->>K2: Load similarities to shared memory
    K2->>K2: Sort and select top-k
    K2->>D: Store top-k indices
    D->>H: Copy results back
```

The data flow follows an asynchronous pattern where data is copied to GPU memory, processed through specialized kernels, and results are copied back to host memory. Each kernel is optimized for specific operations: similarity computation focuses on memory bandwidth utilization while top-k selection emphasizes shared memory sorting efficiency.
