FROM svizor/zoomcamp-model:3.11.5-slim

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy your Flask application
COPY app.py /app/app.py
COPY model1.bin /app/model1.bin

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:9696", "app:app"]
