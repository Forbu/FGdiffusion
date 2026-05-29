FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    networkx \
    poetry \
    pytest \
    matplotlib \
    lightning \
    x-transformers \
    tensorboardX \
    einops \
    heavyball \
    torch_geometric \
    pandas \
    pyarrow

COPY . /app

RUN pip install -e .
