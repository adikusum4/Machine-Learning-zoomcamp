import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Load the saved model and vectorizer
model_file = 'model1.bin'
dv_file = 'dv1.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

# Function to make predictions
def predict(data):
    # Convert input data into a dictionary format that the vectorizer expects
    data_dicts = [data]  # Assuming 'data' is a single dictionary or list of dictionaries
    X = dv.transform(data_dicts)
    
    # Make prediction
    prediction = model.predict(X)
    
    # Apply inverse of log1p transformation (exponentiate to get back to the original scale)
    return np.expm1(prediction)  # Inverse of np.log1p

# Example usage with provided sample_data
sample_data = {
    'state': 'Alabama',
    'county': 'Baldwin',
    'year': 2018,         
    'days_with_aqi': 270,
    'good_days': 245,
    'moderate_days': 25,
    'unhealthy_for_sensitive_groups_days': 0,
    'unhealthy_days': 0,
    'very_unhealthy_days': 0,
    'hazardous_days': 0,
    'max_aqi': 97,
    '90th_percentile_aqi': 50,        
    'median_aqi': 35,
    'days_co': 0,
    'days_no2': 0,
    'days_ozone': 214,
    'days_so2': 0,
    'days_pm2.5': 56,
    'days_pm10': 0,
    'latitude': 30.497478,
    'longitude': -87.880258,
    'totalpopulation': 218022
}

predicted_value = predict(sample_data)
print(f'Predicted value: {predicted_value}')
