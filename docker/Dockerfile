# ==========================================
# STAGE 1: Builder (Compilers & Installation)
# ==========================================
FROM continuumio/miniconda3:latest as builder

WORKDIR /app

# 1. Install Build Dependencies (Heavy tools, discarded later)
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Conda Environment
COPY spherex-pipelines/docker/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# 3. Activate Environment for subsequent commands
# We set the path so 'python' and 'pip' automatically point to the new env
ENV PATH /opt/conda/envs/spxpipe-L3/bin:$PATH

# 4. Copy ALL Source Code (Needed for installation)
COPY spherex-parameters /app/spherex-parameters
COPY simulator /app/simulator
COPY spherex-pipelines /app/spherex-pipelines

# 5. Build and Install
# Note: We use --no-cache-dir to keep the layer slightly smaller,
# though it matters less in the builder stage.
RUN cd /app/spherex-parameters && pip install --no-build-isolation --no-cache-dir . \
    && cd /app/simulator/py \
    && cp /app/spherex-pipelines/tests/external/simulator_external_files.json external_files.json \
    && pip install --no-build-isolation --no-cache-dir . \
    && cd /app/spherex-pipelines \
    && pip install --no-build-isolation --no-cache-dir ".[tractor]"

# 6.Reduce the size of the Conda environment by deleting everything not needed at runtime.
RUN conda clean -afy

# ==========================================
# STAGE 2: Final Runtime (Slim & Clean)
# ==========================================
# Note: Capitalized 'AS' to fix the Docker warning
FROM mambaorg/micromamba:ubuntu22.04 AS final

# 1. Temporarily switch to root to install system packages
USER root
WORKDIR /app
# 2. Install Runtime Libraries ONLY
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the Pre-Built Conda Environment
COPY  --from=builder /opt/conda/envs/spxpipe-L3 /opt/conda/envs/spxpipe-L3

# 5. Setup Environment (Added '=' to fix the legacy key/value warning)
ENV PATH=/opt/conda/envs/spxpipe-L3/bin:$PATH

# 6. Final Setup
RUN echo "source activate spxpipe-L3" >> ~/.bashrc