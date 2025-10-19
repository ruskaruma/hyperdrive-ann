FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install Python dependencies using uv
RUN uv pip install \
    pybind11 \
    numpy \
    pytest

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Create build directory and build
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Test the build
RUN cd build && ./hyperdrive-ann --go

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["./hyperdrive-ann", "--go"]
