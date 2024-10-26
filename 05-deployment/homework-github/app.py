from flask import Flask, request, jsonify
import pickle
import requests  # Import requests

app = Flask(__name__)

# Load model and vectorizer
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request for prediction")  # Log when a request is received
    client = request.get_json()
    print("Client data:", client)  # Log the received data
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return jsonify({'probability': y_pred[0]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
