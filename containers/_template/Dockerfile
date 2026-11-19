# Base image

### Switch
FROM ubuntu:24.04
# FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
###

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
    libfftw3-dev \
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
# RUN /home/xeranes/venv/bin/python -m pip install --no-cache-dir \
#     numpy \
#     scipy \
#     cython \
#     matplotlib \
#     tqdm \
#     numba

# Clone the Espresso repository
# ESPResSo 5.0.1

# There is some BULLSHIT here where the walberla git clone currently (5.10.26) included in ESPResSo 5.0.1 doesn't work, so we need to clone it separately and point to it during the build process. cmake 

# RUN git clone --branch 5.0.1 --depth 1 https://github.com/espressomd/espresso && mkdir build

RUN git clone --branch 5.0.1 --depth 1 \
    https://github.com/espressomd/espresso \
    /home/xeranes/espresso

# Install Python dependencies from requirements.txt
RUN /home/xeranes/venv/bin/python -m pip install --no-cache-dir \
    -r /home/xeranes/espresso/requirements.txt

# waLBerla revision required by ESPResSo 5.0.1
RUN git clone --branch v7.2 --depth 1 \
    https://github.com/lssfau/walberla.git \
    /home/xeranes/walberla


RUN mkdir /home/xeranes/espresso/build

# Build Espresso
COPY myconfig.hpp /home/xeranes/espresso/build/myconfig.hpp
WORKDIR /home/xeranes/espresso/build
# Cmake
# RUN cmake .. -D ESPRESSO_BUILD_WITH_NLOPT=ON && make
RUN cmake .. \
    -D ESPRESSO_BUILD_WITH_NLOPT=ON \
    -DFETCHCONTENT_SOURCE_DIR_WALBERLA=/home/xeranes/walberla \
    && make -j"$(nproc)"


# Cleanup
RUN rm -rf /var/cache/* /tmp/* /var/log/* /usr/share/doc/*
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# TODO !!!
# Set environment variables for Espresso
ENV PYTHONPATH="/home/xeranes/espresso/build/src/python"
ENV ESPRESSOPATH="/home/xeranes/espresso/build/"
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# (Optional) Append environment settings to the bashrc
RUN echo 'export ESPRESSOPATH="/home/xeranes/espresso/build/"' >> /home/xeranes/.bashrc

# Keep the container running
ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
