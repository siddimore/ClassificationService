FROM ubuntu:16.04
MAINTAINER Siddharth More <siddimore@gmail.com>
RUN apt-get update

## Pyton installation ##
RUN apt-get install -y python3.5
RUN apt-get install -y python3-pip
RUN apt-get install -y git


ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-wheel \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    # requirements for keras
    python3-h5py \
    python3-yaml \
    python3-pydot \
    python3-nltk  \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# manually update numpy
RUN pip3 --no-cache-dir install -U numpy==1.13.3

ARG TENSORFLOW_VERSION=1.4.1
ARG TENSORFLOW_DEVICE=cpu
ARG TENSORFLOW_APPEND=
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.1.2
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}


RUN pip3 install nltk && \
    mkdir ~/nltk_data && \
    mkdir ~/nltk_data/chunkers && \
    mkdir ~/nltk_data/corpora && \
    mkdir ~/nltk_data/taggers && \
    mkdir ~/nltk_data/tokenizers && \
    python3 -c "import nltk; nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"


#RUN pip3 install gensim nltk==3.0.1
#RUN python3 -m nltk.downloader -d /usr/share/nltk_data stopwords



# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app/


# Install any needed packages specified in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

WORKDIR /


# Make port 80 available to the world outside this container
EXPOSE 5000


# Run app.py when the container launches
# CMD ["python", "/app/app.py"]
ENTRYPOINT python3 /app/app.py
