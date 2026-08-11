FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# xvfb + Qt/OpenGL runtime libs for headless napari screenshot rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        xvfb \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libfontconfig1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-cursor0 \
        libxrender1 \
        libxi6 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/nnUNet

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Dependency layer: cached as long as pyproject.toml/uv.lock are unchanged
COPY nnUNet/pyproject.toml nnUNet/uv.lock nnUNet/setup.py nnUNet/LICENSE nnUNet/readme.md ./
RUN uv sync --frozen --no-install-project --no-dev

# napari for the optional screenshot rendering (not part of uv.lock)
RUN uv pip install --python .venv/bin/python napari==0.7.0 PyQt5==5.15.11

# Code + model weights (weights live in nnUNet/data/results/, see .dockerignore)
COPY nnUNet/ ./
RUN uv sync --frozen --no-dev

COPY run_inference.py ./run_inference.py

ENV PATH="/app/nnUNet/.venv/bin:${PATH}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    nnUNet_n_proc_DA=1

# Non-root user with passwordless sudo. Build with
#   --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
# so files in mounted volumes are read/writable by both container and host user.
ARG USERNAME=topaneu
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ENV HOME=/home/${USERNAME}

CMD ["/bin/bash"]
