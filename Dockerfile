FROM ubuntu:22.04

# Copy application to the container
COPY . /maia/

# Update the system and install Python
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get clean

# Install libgl1 for cv
RUN apt-get install -y python3-opencv

# Install pip    
RUN apt-get install -y python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install requirements as root
RUN pip3 install -r /maia/requirements.txt

# Expose port
EXPOSE 8080

# Helth check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Create a new user
RUN useradd -ms /bin/bash appuser

# Change ownership of the /maia directory to appuser
RUN chown -R appuser:appuser /maia

# Switch to the new user
USER appuser

# Set the working directory
WORKDIR /maia

CMD ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]