FROM python:3.11.10-slim

ENV PYTHONUNBUFFERED=TRUE

# Install pipenv
RUN pip install pipenv

WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --deploy --system

# Copy application files (train, predict, model)
COPY ["train.py", "predict.py", "obesity-levels-model_catboost.bin", "./"]

# Expose the necessary port
EXPOSE 9696

# Run the Flask app using waitress
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
