FROM bz/gym-mupen64plus:0.0.5
LABEL maintainer "Brian Zier <https://github.com/bzier/>"

RUN apt-get update && \
    apt-get install -y \
        cmake \
        git \
        golang \
        htop \
        libjpeg-dev \
        libsm6 \
        libxrender1 \
        zlib1g-dev

RUN pip install --upgrade pip && \
    pip install \
        "gym[atari]" \
        numpy \
        opencv-python \
        scipy \
        six \
        # Locking in Tensorflow at 1.5 for now due to https://github.com/tensorflow/tensorflow/issues/17411
        tensorflow==1.5 && \
    # Don't know why, but this one wanted to be installed separately
    pip install \ 
        universe

# Expose the port for Tensorboard
EXPOSE 12345

# Copy the current directory to the container working dir
WORKDIR /src/universe-starter-agent
COPY . .
