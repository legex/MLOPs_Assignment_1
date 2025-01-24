FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Command to run the model server
CMD ["python", "app.py"]
