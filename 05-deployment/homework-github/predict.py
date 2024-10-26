import pickle

# Muat DictVectorizer dan LogisticRegression model
with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

with open("model1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Data klien
client = {"job": "management", "duration": 400, "poutcome": "success"}

# Transformasi data klien menggunakan DictVectorizer
X = dv.transform([client])

# Dapatkan probabilitas prediksi
probability = model.predict_proba(X)[0, 1]
print(f"Probability of subscription: {probability}")
