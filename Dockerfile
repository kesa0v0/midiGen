FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install mamba-ssm causal-conv1d --no-binary mamba-ssm,causal-conv1d --no-build-isolation

RUN pip install -r requirements.txt && pip uninstall -y torchao || true

CMD ["/bin/bash"]