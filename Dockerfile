FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]