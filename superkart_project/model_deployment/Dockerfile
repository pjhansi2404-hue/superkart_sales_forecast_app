# Use a Python 3.9 Slim image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and the inference script into the container
COPY best_random_forest_model.joblib .
COPY app.py .

# Expose the port that the application will listen on
EXPOSE 8000

# Define the command to run the application when the container starts
CMD ["python", "app.py"]
