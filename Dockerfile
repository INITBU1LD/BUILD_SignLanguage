# Use a base image with Python installed
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Copy the requirements file into the container
COPY requirements.txt .
# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow Lite runtime
RUN pip install --no-cache-dir tflite-runtime
# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy files from your local repository into the container
COPY . /app

# Expose the port on which Streamlit will run (default is 8501)
EXPOSE 8501

# Set the command to run Streamlit when the container starts
CMD ["streamlit", "run", "streamlit/example.py"]
