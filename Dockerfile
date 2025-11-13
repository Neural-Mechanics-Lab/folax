# Use Python 3.11 as base image (matches project requirements)
FROM python:3.11-slim

# Install system dependencies
# - libglu1-mesa: required for gmsh (as seen in CI workflow)
# - build-essential: for compiling Python packages
# - curl: for installing uv
RUN apt-get update && apt-get install -y \
    libglu1-mesa \
    libgl1 \
    libgl1-mesa-dri \
    libxrender1 \
    libx11-6 \
    libxext6 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    libxfixes3 \
    libxft2 \
    libxinerama1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# The folax repo will be mounted at /workspace/folax
# The .venv will be available at /workspace/folax/.venv
# We'll activate it and run the script from there

# Default command (can be overridden)
CMD ["/bin/bash"]

