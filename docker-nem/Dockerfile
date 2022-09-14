FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04 AS noise-estimation-base
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    build-essential \
    libdc1394-22 \
    libjpeg-turbo-progs \
    libexiv2-dev \
    libgl1-mesa-glx \
    libz-dev \
    libexpat-dev \
    libjpeg-dev \
    pkg-config \
    # locales \
    # language-pack-en \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    python3-pip \
    python3.7 \
    python3.7-dev \
    && rm -rf /var/lib/apt/lists/*

RUN  update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 \
    && update-alternatives  --set python /usr/bin/python3.7 \
#     # && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 \
#     # && update-alternatives  --set python3 /usr/bin/python3.7 \
#     && python -m pip install --upgrade setuptools pip wheel boto3
    && python -m pip install --upgrade setuptools pip wheel boto3

WORKDIR /workspace

# Install Jupyter Lab and useful extensions
RUN pip install jupyterlab
RUN apt-get -y update \
    && apt-get install -y curl \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt install -y nodejs \
    && jupyter lab build --name='Noise Estimation Model' \
    && pip install ipywidgets \
    && pip install jupyterlab-nvdashboard \
    && pip install ipyexperiments
RUN apt-get update && apt-get install -y git-all

ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--port=8888"]

