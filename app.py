from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Dictionnaire des modèles disponibles
available_models = {
    "1": "models/nizar_model.pth",
    "2": "models/anton_model.pth",
    "3": "models/noufel_model.pth",
    "4": "models/sivappiryan_model.pth"
}

# Chargement des encodeurs communs
label_encoders = joblib.load("models/label_encoders.pth")


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Récupération du modèle sélectionné
        model_id = request.args.get("model_id", type=str)
        if model_id not in available_models:
            return jsonify({"error": f"Invalid model_id: {model_id}. Available model IDs: {list(available_models.keys())}"}), 400
        
        # Chargement dynamique du modèle correspondant
        model_path = available_models[model_id]
        model = joblib.load(model_path)

        # Mapping pour convertir les noms complets en abréviations
        embarked_mapping = {
            "Southampton": "S",
            "Cherbourg": "C",
            "Queenstown": "Q"
        }

        # Récupération des paramètres depuis les query parameters
        sex = request.args.get("sex", type=str)
        age = request.args.get("age", type=float)
        fare = request.args.get("fare", type=float)
        pclass = request.args.get("class", type=str)
        sibsp = request.args.get("sibsp", type=int)
        parch = request.args.get("parch", type=int)
        embarked = request.args.get("embarked", type=str)

        # Conversion du nom complet d'embarked en abréviation
        if embarked in embarked_mapping:
            embarked = embarked_mapping[embarked]

        # Vérification et encodage des paramètres catégoriques
        try:
            sex_encoded = label_encoders["sex"].transform([sex])[0]
        except ValueError:
            return jsonify({"error": f"Invalid value for 'sex': {sex}. Expected values: {label_encoders['sex'].classes_.tolist()}"}), 400

        try:
            class_encoded = label_encoders["class"].transform([pclass])[0]
        except ValueError:
            return jsonify({"error": f"Invalid value for 'class': {pclass}. Expected values: {label_encoders['class'].classes_.tolist()}"}), 400

        try:
            embarked_encoded = label_encoders["embarked"].transform([embarked])[0]
        except ValueError:
            return jsonify({"error": f"Invalid value for 'embarked': {embarked}. Expected values: {label_encoders['embarked'].classes_.tolist()}"}), 400

        # Préparation des données pour la prédiction
        input_data = np.array([[sex_encoded, age, fare, class_encoded, sibsp, parch, embarked_encoded]])

        # Prédiction des probabilités
        probabilities = model.predict_proba(input_data)[0]
        not_survived_prob = probabilities[0] * 100  # Pourcentage pour "Pas survécu"
        survived_prob = probabilities[1] * 100  # Pourcentage pour "Survécu"

        # Résultat final
        prediction = "Survécu" if survived_prob > not_survived_prob else "Pas survécu"

        return jsonify({
            "model_id": model_id,
            "prediction": prediction,
            "probabilities": {
                "not_survived": f"{not_survived_prob:.2f}%",
                "survived": f"{survived_prob:.2f}%"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
