FROM python:3.11-slim

LABEL maintainer="Peter Zhou <comp-psych-org>"
LABEL description="Computational Psychiatry Starter Kit — one-command setup for undergrad researchers"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by scientific packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libatlas-base-dev \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python scientific stack in dependency order
# Core numerics
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.2

# Visualization
RUN pip install --no-cache-dir \
    matplotlib==3.9.0 \
    seaborn==0.13.2

# Statistics & ML
RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    statsmodels==0.14.2

# Bayesian inference
RUN pip install --no-cache-dir \
    pymc==5.15.1 \
    arviz==0.18.0

# Neuroimaging
RUN pip install --no-cache-dir \
    mne==1.7.0 \
    nilearn==0.10.4

# JupyterLab
RUN pip install --no-cache-dir \
    jupyterlab==4.2.2 \
    jupytext==1.16.2

# NOTE: hBayesDM and CmdStan are intentionally excluded.
# They require a C++ toolchain and ~2GB of Stan binaries,
# which makes the Docker image impractically large.
# See README.md for optional local installation instructions.

# Create working directory
WORKDIR /home/comp-psych

# Copy tutorial notebooks and example data
COPY notebooks/ ./notebooks/
COPY data/ ./data/

# Configure Jupytext to pair .py files with .ipynb on open
RUN mkdir -p /root/.jupyter && \
    echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"' \
    > /root/.jupyter/jupyter_server_config.py

# Expose JupyterLab port
EXPOSE 8888

# Launch JupyterLab with no token/password for local dev convenience
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
