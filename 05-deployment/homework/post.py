import requests

url = "https://urban-space-capybara-4j79ggg5vj6wfjpxp-9696.app.github.dev/predict"
client = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=client)

# Periksa status kode respons
print("Status Code:", response.status_code)
print("Response Text:", response.text)

# Jika status kode adalah 200, coba ambil JSON
if response.status_code == 200:
    print(response.json())
else:
    print("Request failed")
