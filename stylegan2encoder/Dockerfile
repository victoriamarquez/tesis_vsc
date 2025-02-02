# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html


FROM tensorflow/tensorflow:1.15.0-gpu-py3

# Install required Python packages
RUN pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 keras==2.3.1

# Add your repository source and allow unauthenticated packages
RUN echo 'deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ /' > /etc/apt/sources.list.d/untrusted.list

RUN cat /etc/apt/sources.list | echo
# Install dependencies for dlib and other system packages
RUN apt-get --allow-unauthenticated update && apt-get --allow-unauthenticated install -y \
    cmake \
    wget \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dlib
RUN pip install dlib

# Install any other Python dependencies you might need for your project
# Here you can add more pip install commands if needed
