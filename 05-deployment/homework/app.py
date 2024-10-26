import pickle
from flask import Flask, request, jsonify

# Inisialisasi Flask app
app = Flask(__name__)

# Muat model dan vectorizer
with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

with open("model1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Buat endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    probability = model.predict_proba(X)[0, 1]
    return jsonify({"probability": probability})

@app.route('/')
def home():
    return "Welcome to the Flask app!"

#@app.route('/api/some_endpoint', methods=['GET'])
#def some_endpoint():
#    return jsonify({"message": "This is a response from some_endpoint!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
