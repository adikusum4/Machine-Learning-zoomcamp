import requests
url = 'http://localhost:9696/predict'

patient = {'gender' : 'Male', 
           'age' : 18.0, 
           'height' : 1.67, 
           'weight' : 53.0, 
           'family_history_overweight' : 'yes', 
           'high_caloric_food' : 'yes', 
           'vegetable_consumption' : 2.0, 
           'main_meals_per_day' : 2.0, 
           'snacking_between_meals' : 'Frequently', 
           'smoking' : 'yes', 
           'water_intake' : 3.0, 
           'calorie_monitoring' : 'no', 
           'physical_activity_frequency' : 1.0, 
           'tech_usage_hours' : 0.0, 
           'alcohol_consumption' : 'yes', 
           'transportation_mode' : 'Automobile', 
           #'obesity_level': 'Normal_Weight'
           }


print("Sending request to the server...")

try:
    response = requests.post(url, json=patient)
    print("Request sent successfully!")
    print("Response status:", response.status_code)

    if response.status_code == 200:
        # If successful, display the JSON response
        print("Response body:", response.json())
    else:
        # If it fails, display the error message from the server
        print("Server returned an error:", response.text)
except requests.exceptions.RequestException as e:
    # If the server does not respond or there is a connection issue
    print("An error occurred while sending the request:", str(e))

