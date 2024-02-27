# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10
WORKDIR /app 
COPY . /app
RUN pip install -U pip setuptools wheel && pip install -U --no-cache-dir -e .

# Make port available to the world outside this container, if needed
# EXPOSE 8000

# Run llm_correctness.py when the container launches
# CMD ["python", "./llm_correctness.py"]
