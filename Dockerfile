# Base image
FROM ubuntu:22.04

# Set non-interactive frontend to suppress prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-venv \
    python3-dev \
    python3-pip \
    openmpi-bin \
    libboost-all-dev \
    fftw3-dev \
    libfftw3-mpi-dev \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/xeranes

# Create and activate a virtual environment
RUN python3 -m venv /home/xeranes/venv && \
    /home/xeranes/venv/bin/pip install --upgrade pip

# Add the virtual environment to PATH
ENV PATH="/home/xeranes/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/xeranes/venv"

# Install Python packages in the virtual environment
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    cython 

# Clone the custom Espresso repository
RUN git clone --branch egg_model_andrey_copy https://github.com/stekajack/espresso_patched.git espresso

COPY configs/egg_cfg.hpp /home/xeranes/espresso/build/myconfig.hpp
COPY patches/small_tweak.patch /home/xeranes/patches/small_tweak.patch

# Build Espresso
WORKDIR /home/xeranes/espresso
# RUN git apply /home/xeranes/patches/small_tweak.patch

RUN cd build && \
    cmake .. -DESPRESSO_BUILD_WITH_CUDA=OFF && \
    make
#   make -j$(nproc)

# Clone and Install Pressomancy
WORKDIR /home/xeranes
RUN git clone --branch main https://github.com/stekajack/pressomancy.git && \
    cd pressomancy && \
    pip install -e .
RUN mkdir DATA_VIEW
RUN mkdir UPLOAD_VIEW

# Cleanup
RUN rm -rf /var/cache/* /tmp/* /var/log/* /usr/share/doc/*
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Espresso
ENV PYTHONPATH="/home/xeranes/espresso/build/src/python"
ENV ESPRESSOPATH="/home/xeranes/espresso/build/"
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# (Optional) Append environment settings to the bashrc
RUN echo 'export ESPRESSOPATH="/home/xeranes/espresso/build/"' >> /home/xeranes/.bashrc

# Keep the container running
ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
