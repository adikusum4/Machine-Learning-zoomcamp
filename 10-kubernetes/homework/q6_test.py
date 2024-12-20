import time
import requests


url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

while True:
    time.sleep(0.1) #sleep(0.1)  # Adjust the frequency if necessary
    response = requests.post(url, json=client).json()
    print(response)

#print(response)
