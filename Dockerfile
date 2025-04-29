#  Choose a specific version
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY vulnerable_ml_model.py .

# Copy any necessary data files (e.g., data.csv, model.pkl) if you were using any.
# For example, if you create a malicious pickle file as part of your demonstration
#COPY malicious_model.pkl .

# Install the required Python libraries (dependencies)
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
# Command to run your Python script.  This will execute when the container starts.
CMD ["python", "vulnerable_ml_model.py"]