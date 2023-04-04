# Use the PyTorch base image with CUDA and cuDNN
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt-get update && apt-get install -y git

# Clone LMFlow from GitHub
RUN git clone https://github.com/OptimalScale/LMFlow.git

# Install necessary dependencies
RUN apt-get update && apt-get install -y build-essential
RUN conda create -n lmflow python=3.9 -y
SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN echo "conda activate lmflow" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install mpi4py
RUN pip install -e ./LMFlow
RUN pip install jsonlines

# Set the default command to activate the lmflow conda environment
CMD ["conda", "activate", "lmflow"]
