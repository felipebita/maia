# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \

    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash appuser

# Create app directory
WORKDIR /maia

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Change ownership of the /maia directory to appuser
RUN chown -R appuser:appuser /maia

# Switch to the new user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# Run the application
CMD ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
