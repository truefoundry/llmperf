# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10
WORKDIR /app 
RUN apt update && \
    apt install -y --no-install-recommends jq && \
    pip install -U pip setuptools wheel && \
    pip install -U --no-cache-dir yq
COPY pyproject.toml /app/
RUN pip install $(tomlq .project.dependencies[]  pyproject.toml | xargs)
COPY . /app
RUN pip install --no-cache-dir -e .
# Make port available to the world outside this container, if needed
# EXPOSE 8000

# Run llm_correctness.py when the container launches
# CMD ["python", "./llm_correctness.py"]
