FROM debian:buster

WORKDIR /app

ENV PATH="${PATH}:/miniconda/bin" \
REPO_PATH=/app/epos/repository \
STORE_PATH=/app/epos/store \
BOP_PATH=/app/bop/datasets

ENV TF_DATA_PATH=$STORE_PATH/tf_data \
TF_MODELS_PATH=$STORE_PATH/tf_models \
PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_renderer/build:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/progressive-x/build:$REPO_PATH/external/slim:${PYTHONPATH} \
LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:${LD_LIBRARY_PATH} \
OSMESA_PREFIX=$REPO_PATH/external/osmesa \
LLVM_PREFIX=$REPO_PATH/external/llvm \
CMAKE_PREFIX_PATH=/miniconda/envs/epos \
PYTHON_PREFIX=/miniconda/envs/epos

RUN apt-get update && apt-get install --yes wget build-essential libeigen3-dev curl autoconf cmake automake libtool pkg-config zlib1g-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda &&\
    rm Miniconda3-latest-Linux-x86_64.sh

COPY . $REPO_PATH/

RUN conda install mamba -n base -c conda-forge
RUN mamba env create --file=$REPO_PATH/environment.yml
RUN mamba clean --all -y

RUN cd $REPO_PATH/external/progressive-x/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

RUN mkdir $OSMESA_PREFIX -p && \
    mkdir $LLVM_PREFIX -p && \
    cd $REPO_PATH/external/bop_renderer/osmesa-install && \
    mkdir build && cd build && \
    MKJOBS=$(nproc) /bin/bash ../osmesa-install.sh

RUN cd $REPO_PATH/external/bop_renderer && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

RUN conda init bash
RUN echo "conda activate epos" >> ~/.bashrc
