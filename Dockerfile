# Base image
FROM python:3.8.8-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose application port (change if needed)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
