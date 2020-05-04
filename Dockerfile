FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN curl -sL https://deb.nodesource.com/setup_14.x | -E bash -
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  vim \
  wget \
  openssh-server \
  mysql-client \
  ca-certificates \
  unzip \
  rsync \
  nodejs \
  psmisc && \
  rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
  bash ./miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:$PATH"
RUN conda config --add channels conda-forge && \
  conda update conda && \
  conda update --all

COPY environment.yml /root
RUN conda env update --name base -f /root/environment.yml

WORKDIR /
# WORKDIR /root/zg-p1
# COPY . .
