import requests
#url = "http://EstimationObesityLevels.us-east-1.elasticbeanstalk.com/predict"
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
           #'obesity_level
           #'obesity_level': 'Normal_Weight'
           }

print(requests.post(url, json=patient).json())

#response = requests.post(url, json=home).json()
#print()

#print('The estimated value of the home is ${:,.2f}'.format(response['home_price']))

#pipenv install numpy pandas scikit-learn flask catboost waitress requests cloudpickle
#pipenv install awsebcli --dev