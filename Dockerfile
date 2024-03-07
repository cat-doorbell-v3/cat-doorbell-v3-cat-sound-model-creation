# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /output

WORKDIR /app

COPY docker /app

# Install necessary packages including Python3 and tools for Bazel
RUN apt-get update -qq && apt-get install -y -qq \
    git \
    curl \
    g++ \
    unzip \
    zip \
    gnupg \
    python3 \
    python3-pip \
    wget \
    ffmpeg

RUN pip3 install --quiet --quiet --no-cache-dir -r requirements.txt

WORKDIR /input
RUN wget --quiet http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz \
    && tar -xzf features.tar.gz \
    && mv audioset_v1_embeddings audioset

# Remove the tar file after extraction
RUN rm features.tar.gz

WORKDIR /app

CMD python3 extract_samples.py
