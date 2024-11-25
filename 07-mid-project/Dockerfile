# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in Pipfile
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Expose port 5000
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app when the container launches
CMD ["pipenv", "run", "flask", "run"]
