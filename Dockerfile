# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements-dev.txt
RUN pip install .

# Make port available to the world outside this container, if needed
# EXPOSE 8000

# Define environment variable, if needed
# ENV NAME Value

# Run llm_correctness.py when the container launches
# CMD ["python", "./llm_correctness.py"]