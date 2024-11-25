# Gunakan Python resmi
FROM python:3.9-slim

# Install necessary dependencies
RUN pip install numpy pandas 
RUN pip install scikit-learn

# Set direktori kerja dalam container
WORKDIR /app

# Salin Pipfile dan Pipfile.lock ke container
COPY Pipfile Pipfile.lock ./

# Instal Pipenv dan dependensi
RUN pip install --no-cache-dir pipenv && pipenv install --system --deploy

# Salin semua file proyek ke dalam container
COPY . .

# Ekspose port (untuk Flask, jika diperlukan)
EXPOSE 5000

# Perintah default (ubah sesuai kebutuhan)
CMD ["python", "train.py"]
