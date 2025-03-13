# Use an official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Expose port (if your AI has an API, e.g., using Flask)
EXPOSE 5000

# Define the command to run your AI script
CMD ["python", "your_script.py"]

