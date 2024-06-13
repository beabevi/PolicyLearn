# Build with
# docker build --build-arg GH_ACCESS_TOKEN=<REPLACEME> --force-rm -t policy-learn -f Dockerfile .

# Run interactively
# docker run --runtime=nvidia --gpus all --rm -it policy-learn
# inside the container# git pull # (optionally update to latest version from github)
# inside the container#  WANDB_API_KEY=<REPLACEME> WANDB_ENTITY=<REPLACEME> python train.py batch_size=128 gumbel_softmax_tau=0.33 num_marked=1 num_subgraphs=3 opt.num_epochs=400 seed=2 selection_opt.num_epochs=400 selection_opt.optimizer.lr=0.001 selection_type=gumbel task=ZINC 'model_backbone.GNNConv=${callable:models.CustomGINE}' 'selection_backbone.GNNConv=${callable:models.CustomGINE}' optim@opt.scheduler=ReduceLROnPlateauConfig


ARG CUDA_VERSION=11.8.0
ARG CUDA_TAG=${CUDA_VERSION}-base-ubuntu22.04

FROM nvcr.io/nvidia/cuda:${CUDA_TAG}

ARG GH_ACCESS_TOKEN

# cuda-nvcc-11-8 needed for compiling custom torch C++/CUDA extensions
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y wget git g++ cuda-nvcc-11-8 && apt clean

WORKDIR /workdir
SHELL ["/bin/bash", "-c"]

# Install Miniconda.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
 && chmod +x miniconda.sh \
 && bash miniconda.sh -b -p /workdir/miniconda \
 && rm /workdir/miniconda.sh
ENV PATH=/workdir/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create the environment
RUN /workdir/miniconda/bin/conda install conda-build \
 && /workdir/miniconda/bin/conda create -y --name my-env python=3.10.9 \
 && /workdir/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=my-env
ENV CONDA_PREFIX=/workdir/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install pytorch
RUN conda install -y numpy pytorch pytorch-cuda=11.8 -c pytorch -c nvidia && /workdir/miniconda/bin/conda clean -ya

# Install pyg
RUN conda install -y pyg pytorch-scatter -c pyg && /workdir/miniconda/bin/conda clean -ya

# Install other utilities
RUN pip install hydra-core ogb wandb torchmetrics --upgrade

RUN git clone https://oauth2:$GH_ACCESS_TOKEN@github.com/beabevi/policy-learn.git
WORKDIR /workdir/policy-learn

# Compile extensions
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6" pip install -e .
