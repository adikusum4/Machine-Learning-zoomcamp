FROM svizor/zoomcamp-model:3.11.5-slim

# Install dependencies dari Pipenv
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --system --deploy

# Salin script Flask Anda ke container
COPY app.py /app/app.py

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]
