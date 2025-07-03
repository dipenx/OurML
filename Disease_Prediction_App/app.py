from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

### ----------------------------- Stress Detection ----------------------------- ###

# Load the stress detection model
STRESS_MODEL_PATH = 'stress_model.pkl'

def load_stress_model(model_path: str):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print(f"⚠️ Stress model not found at {model_path}. Please train it first.")
        return None

stress_model = load_stress_model(STRESS_MODEL_PATH)
@app.route("/stress", methods=['GET', 'POST'])
def stress_index():
    
    result = None
    if request.method == 'POST':
        try:
            features = [
                int(request.form.get('overwhelmed', 0)),
                int(request.form.get('sleep', 0)),
                int(request.form.get('mood', 0)),
                int(request.form.get('work_hours', 0)),
                int(request.form.get('concentration', 0)),
                int(request.form.get('fatigue', 0)),
                int(request.form.get('anxiety', 0)),
                int(request.form.get('isolation', 0)),
                int(request.form.get('appetite', 0)),
                int(request.form.get('activity', 0))
            ]

            if stress_model:
                prediction = stress_model.predict([features])[0]
                result = {
                    0: "🟢 Low Stress",
                    1: "🟡 Moderate Stress",
                    2: "🔴 High Stress"
                }.get(prediction, "Unknown Stress Level")
            else:
                result = "⚠️ Stress model not available. Please train it first."
        except Exception as e:
            result = f"❌ Error: {str(e)}"

    return render_template("stress.html", result=result)


### ----------------------------- Disease Prediction ----------------------------- ###

# Load disease model and encoder
try:
    disease_model = joblib.load("disease_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"⚠️ Error loading disease model: {e}")
    disease_model = None
    label_encoder = None

# Load symptoms list
try:
    data = pd.read_csv("dataset/Training.csv")
    symptoms_list = list(data.columns[:-1])  # Exclude 'prognosis'
except Exception as e:
    print(f"⚠️ Error loading symptoms: {e}")
    symptoms_list = []

# Load home remedies
try:
    remedies_df = pd.read_csv("dataset/home_remedies.csv")
    home_remedies_dict = dict(zip(remedies_df["Disease"].str.lower(), remedies_df["HomeRemedies"]))
except Exception as e:
    print(f"⚠️ Error loading home remedies: {e}")
    home_remedies_dict = {}

# Load descriptions
try:
    descriptions_df = pd.read_csv("dataset/disease_descriptions.csv")
    disease_description_dict = dict(zip(descriptions_df["Disease"].str.lower(), descriptions_df["Description"]))
except Exception as e:
    print(f"⚠️ Error loading disease descriptions: {e}")
    disease_description_dict = {}

# Load precautions
try:
    precautions_df = pd.read_csv("dataset/disease_precautions.csv")
    disease_precautions_dict = {
        row["Disease"].strip().lower(): [
            row["Precaution_1"], row["Precaution_2"],
            row["Precaution_3"], row["Precaution_4"]
        ]
        for _, row in precautions_df.iterrows()
    }
except Exception as e:
    print(f"⚠️ Error loading disease precautions: {e}")
    disease_precautions_dict = {}




# Load the medicine dataset
data = pd.read_csv("dataset/medications.csv")
disease_med_dict = dict(zip(data["Disease"].str.lower(), data["Medication"]))

@app.route("/medicine")
def medicine():
    return render_template("medicine.html")

@app.route('/get_diseases')
def get_diseases():
    return jsonify(list(disease_med_dict.keys()))

@app.route('/get_medication', methods=["POST"])
def get_medication():
    disease = request.json.get("disease", "").strip().lower()
    medication = disease_med_dict.get(disease, "No medication found for this disease.")
    return jsonify({"medication": medication})



@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not disease_model or not label_encoder:
        return jsonify({"error": "❌ Model is not loaded properly. Please check your model files."})

    try:
        input_data = request.json
        symptoms = input_data.get("symptoms", "").split(",")

        # Normalize
        symptoms = [symptom.strip().lower() for symptom in symptoms]

        input_features = np.zeros(len(symptoms_list))
        matched = False

        for i, symptom in enumerate(symptoms_list):
            if symptom.lower() in symptoms:
                input_features[i] = 1
                matched = True

        if not matched:
            return jsonify({"error": "⚠️ No valid symptoms provided."})

        prediction = disease_model.predict([input_features])[0]
        disease_name = label_encoder.inverse_transform([prediction])[0]

        remedies = home_remedies_dict.get(disease_name.lower(), "❌ No home remedies found.")
        description = disease_description_dict.get(disease_name.lower(), "❌ No description available.")
        precautions = disease_precautions_dict.get(disease_name.lower(), [])

        return jsonify({
            "disease": disease_name,
            "description": description,
            "home_remedies": remedies,
            "precautions": precautions
        })

    except Exception as e:
        return jsonify({"error": f"❌ An error occurred: {str(e)}"})

@app.route("/get_symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptoms_list})


if __name__ == "__main__":
    app.run(debug=True)
