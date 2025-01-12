# Use a base image with Python
FROM python:3.12-slim

RUN pip install -r requirements.txt

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (if any required for specific libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv to manage dependencies
RUN pip install --no-cache-dir pipenv

# Install Python dependencies from Pipfile.lock (if it exists) or Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Expose the port that your application will run on for Jupyter and Gunicorn
EXPOSE 8888
EXPOSE 9696

# Set default command to run Jupyter Notebook (optional, remove if only API needed)
CMD ["pipenv", "run", "jupyter", "notebook", "--no-browser", "--ip=0.0.0.0"]

# Use gunicorn to serve the Flask app in predict.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
