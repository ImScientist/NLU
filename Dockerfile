FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv pip install --system -r requirements.txt

# Expose JupyterLab port
EXPOSE 8888

# Run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
