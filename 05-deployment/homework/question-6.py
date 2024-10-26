import requests

url = "http://localhost:9696/predict"  # Update this to the correct endpoint
client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client)

# Check the response
print(response.status_code)  # This should be 200 if successful
print(response.json())  # This will print the response containing the probability
