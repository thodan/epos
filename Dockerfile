FROM ubuntu:bionic
WORKDIR /app
RUN apt-get update && apt-get install --yes wget build-essential libeigen3-dev curl autoconf cmake automake libtool pkg-config zlib1g-dev git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/miniconda/bin:${PATH}"

RUN conda create -y --name epos python=3.6 anaconda

SHELL ["conda", "run", "-n", "epos", "/bin/bash", "-c"]

RUN conda install numpy=1.16.6 -y && \
    conda install tensorflow-gpu=1.12.0 -y && \
    conda install pyyaml=5.3.1 -y && \
    conda install opencv=3.4.2 -y && \
    conda install pandas=1.0.5 -y && \
    conda install tabulate=0.8.3 -y && \
    conda install imageio=2.9.0 -y && \
    conda install -c mjirik pypng=0.0.18 -y && \
    conda install -c conda-forge igl -y && \
    conda install glog=0.4.0 -y

SHELL ["/bin/sh", "-c"]

ENV REPO_PATH=/app/epos/repository \
STORE_PATH=/app/epos/store \
BOP_PATH=/app/bop/datasets \
TF_DATA_PATH=$STORE_PATH/tf_data \
TF_MODELS_PATH=$STORE_PATH/tf_models \
PYTHONPATH=$REPO_PATH:$PYTHONPATH \
PYTHONPATH=$REPO_PATH/external/bop_renderer/build:$PYTHONPATH \
PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH \
PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH \
PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH \
LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:$LD_LIBRARY_PATH \
OSMESA_PREFIX=$REPO_PATH/external/osmesa \
LLVM_PREFIX=$REPO_PATH/external/llvm \
CMAKE_PREFIX_PATH=/miniconda/envs/epos \
PYTHON_PREFIX=/miniconda/envs/epos

RUN git clone --recurse-submodules https://github.com/thodan/epos.git $REPO_PATH/

RUN mkdir $OSMESA_PREFIX -p &&\
    mkdir $LLVM_PREFIX -p &&\
    cd $REPO_PATH/external/bop_renderer/osmesa-install &&\
    mkdir build && cd build &&\
    /bin/bash ../osmesa-install.sh

RUN cd $REPO_PATH/external/bop_renderer &&\
    mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make -j16

RUN cd $REPO_PATH/external/progressive-x/build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make -j16