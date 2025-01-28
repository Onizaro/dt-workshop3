from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Load the Titanic dataset
from seaborn import load_dataset
titanic = load_dataset('titanic')

# Preprocess the dataset
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
titanic = titanic.dropna()

# Encode categorical features
label_encoder = LabelEncoder()
titanic['sex'] = label_encoder.fit_transform(titanic['sex'])

# Define features and target
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = titanic['survived']

# Train the Decision Tree model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Enregistrement du modèle et des encodeurs
joblib.dump(model, "models/noufel_model.pth")
joblib.dump(label_encoder, "models/label_encoders.pth")
print("Modèle et encodeurs sauvegardés avec succès !")

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Titanic Decision Tree API is running!"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extraire les paramètres de la requête
        pclass = int(request.args.get('pclass'))
        sex = request.args.get('sex')
        age = float(request.args.get('age'))
        sibsp = int(request.args.get('sibsp'))
        parch = int(request.args.get('parch'))
        fare = float(request.args.get('fare'))
        
        # Encoder 'sex'
        sex_encoded = label_encoder.transform([sex])[0]
        
        # Créer le tableau d'entrée
        input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        predicted_survival = "Survived" if prediction[0] == 1 else "Did not survive"

        # Calculer les probabilités
        probas = model.predict_proba(input_data)
        
        # Probabilités pour "not_survived" (0) et "survived" (1)
        prob_not_survived = probas[0][0] * 100
        prob_survived = probas[0][1] * 100

        # Formater la réponse
        response = {
            "input": {
                "pclass": pclass,
                "sex": sex,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare
            },
            "prediction": predicted_survival,  # Utiliser 'predicted_survival' au lieu de 'prediction'
            "probabilities": {
                "not_survived": f"{prob_not_survived:.2f}%",
                "survived": f"{prob_survived:.2f}%"
            },
            "status": "success"
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
