import requests

url = "https://urban-space-capybara-4j79ggg5vj6wfjpxp-9696.app.github.dev/predict"
client = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=client)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")  # Print the raw response text

try:
    print(response.json())
except ValueError as e:
    print("Error parsing JSON:", e)
