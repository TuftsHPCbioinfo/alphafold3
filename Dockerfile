# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Some RUN statements are combined together to make Docker build run faster.
# Get latest package listing, install python, git, wget, compilers and libs.
# * git is required for cloning AlphaFold 3 and for pyproject.toml toolchain.
# * gcc, g++, make are required for compiling HMMER and AlphaFold 3 libraries.
# * zlib is a required dependency of AlphaFold 3.
RUN DEBIAN_FRONTEND=noninteractive \
apt-get update --quiet \
&& apt-get install --yes --quiet python3.12 python3.12-dev \
&& apt-get install --yes --quiet git wget gcc g++ make zlib1g-dev zstd

# Install uv from the official repository. The version is pinned for
# reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

# UV_COMPILE_BYTECODE=1 speeds up future container starts.
# UV_PROJECT_ENVIRONMENT explicitly sets the virtual environment location.
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/alphafold3_venv
RUN uv venv $UV_PROJECT_ENVIRONMENT

ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"

# Clone AlphaFold 3 v3.0.3 source code. Doing this before building HMMER means
# the jackhmmer_seq_limit.patch is available without a separate COPY, and the
# build no longer requires any local source files in the Docker build context.
RUN git clone --branch v3.0.3 --depth 1 \
    https://github.com/google-deepmind/alphafold3.git /app/alphafold

# Install HMMER. Do so before the heavy uv sync step so Docker can cache the
# image layer containing HMMER. The sequence limit patch is pulled directly from
# the cloned repo rather than from the local build context.
# Note: eddylab.org doesn't support HTTPS and the GitHub tar is explicitly not
# recommended for building from source.

# Download, check hash, and extract the HMMER source code.
RUN mkdir /hmmer_build /hmmer ; \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build ; \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz)

# Apply the --seq_limit patch to HMMER (sourced from the cloned repo).
RUN (cd /hmmer_build && patch -p0 < /app/alphafold/docker/jackhmmer_seq_limit.patch)

# Build HMMER.
RUN (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j) && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -R /hmmer_build

WORKDIR /app/alphafold

# Add shebang line to run_*.py scripts and make them executable so they can be
# invoked directly (e.g. `run_alphafold.py --help`) without prefixing python3.
RUN for script in /app/alphafold/run_*.py; do \
        [ -f "$script" ] || continue; \
        sed -i '1s|^|#!/usr/bin/env python3\n|' "$script"; \
        chmod +x "$script"; \
    done

# Add /app/alphafold to PATH so run_*.py scripts are findable without a path prefix.
ENV PATH="/app/alphafold:$PATH"

# Install the exact dependency tree using uv and cache the build artifacts.
# --frozen: do not update the lockfile during build.
# --all-groups: install development/test dependencies defined in pyproject.toml.
# --no-editable: install as a static package.
# If using this as a recipe for local installation, we recommend removing the
# --frozen and --no-editable flags.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --all-groups --no-editable

# Build chemical components database (this binary was installed by uv sync).
RUN uv run build_data

# To work around a known XLA issue causing the compilation time to greatly
# increase, the following environment variable setting XLA flags must be enabled
# when running AlphaFold 3. Note that if using CUDA capability 7 GPUs, it is
# necessary to set the following XLA_FLAGS value instead:
# ENV XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
# (no need to disable gemm in that case as it is not supported for such GPU).
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# Memory settings used for folding up to 5,120 tokens on A100 80 GB.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["run_alphafold.py"]

