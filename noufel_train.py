from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

import seaborn as sns

titanic = sns.load_dataset("titanic")
data = titanic[["sex", "age", "fare", "class", "sibsp", "parch", "embarked", "survived"]]
data = data.dropna()

# Encodage des variables catégoriques
label_encoders = {}
for col in ["sex", "class", "embarked"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


X = data.drop("survived", axis=1)
y = data["survived"]

# Train the Decision Tree model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Enregistrement du modèle et des encodeurs
joblib.dump(model, "models/noufel_model.pth")
joblib.dump(label_encoders, "models/label_encoders.pth")
print("Modèle et encodeurs sauvegardés avec succès !")

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Titanic Decision Tree API is running!"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract query parameters
        pclass = int(request.args.get('pclass'))
        sex = request.args.get('sex')
        age = float(request.args.get('age'))
        sibsp = int(request.args.get('sibsp'))
        parch = int(request.args.get('parch'))
        fare = float(request.args.get('fare'))
        
        # Encode 'sex' using its specific encoder
        sex_encoded = label_encoders["sex"].transform([sex])[0]
        
        # Create input array
        input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_survival = "Survived" if prediction[0] == 1 else "Did not survive"

        # Standardized response format
        response = {
            "input": {
                "pclass": pclass,
                "sex": sex,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare
            },
            "prediction": {
                "survival": predicted_survival,
                "survival_code": int(prediction[0])
            },
            "status": "success"
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
