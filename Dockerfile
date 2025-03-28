# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt /app/requirements.txt

# Install Python dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . /app

# Expose the port that the application runs on.
EXPOSE 5000

# Run the Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port", "5000"]




# A few notes on this Dockerfile:
# Base Image: We are using a lightweight Python image (python:3.9-slim).
# Working Directory: The /app directory in the container will hold your application files.
# Dependency Installation: We copy requirements.txt first, to leverage Docker’s build cache, and then install the Python packages.
# Port Exposure: Streamlit by default runs on port 8501. We expose that port.
# CMD: The command starts the Streamlit app. Here, we rely on the environment variables we set for port and address, but you can also pass them as arguments if needed.