import cloudpickle
import pandas as pd
import numpy as np
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)

def predict_level(patient, pipe, le, final_catboost):
    patient = pd.DataFrame(data=patient, index=[0])
    X_test = pipe.transform(patient)
    pred = final_catboost.predict(X_test)
    return le.inverse_transform(np.ravel(pred))[0]

with open('obesity-levels-model_catboost.bin', 'rb') as f_in:
    pipe, le, final_catboost = cloudpickle.load(f_in)

print("Model, pipeline, and label encoder loaded successfully.")

app = Flask('estimation-obesity-levels')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient = request.get_json()
        if not patient or 'gender' not in patient or 'age' not in patient or 'weight' not in patient:
            result = {'error': 'Invalid input: Missing required fields'}
            return jsonify(result), 400  # Bad Request

        logging.info(f"Received request: {patient}")
        prediction = predict_level(patient, pipe, le, final_catboost)
        logging.info(f"Prediction Result: {prediction}")
        result = {'obesity_level': str(prediction)}
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        result = {'error': str(e)}
        return jsonify(result), 500  # Internal Server Error
    return jsonify(result)

@app.route('/')
def index():
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
