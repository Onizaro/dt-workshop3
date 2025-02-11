from flask import Flask, request, jsonify
import joblib
import numpy as np

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

def preprocess_input(args):
    """Prépare les données pour la prédiction."""
    try:
        sex = label_encoders["sex"].transform([args.get("sex")])[0]
        age = float(args.get("age"))
        fare = float(args.get("fare"))
        pclass = label_encoders["class"].transform([args.get("class")])[0]
        sibsp = int(args.get("sibsp"))
        parch = int(args.get("parch"))
        
        embarked_mapping = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
        embarked = embarked_mapping.get(args.get("embarked"), args.get("embarked"))
        embarked = label_encoders["embarked"].transform([embarked])[0]

        return np.array([[sex, age, fare, pclass, sibsp, parch, embarked]])
    
    except ValueError as e:
        return {"error": str(e)}

# Ajout du dictionnaire des poids des modèles
model_weights = {model_id: 1.0 for model_id in available_models}

@app.route("/predict", methods=["GET"])
def predict():
    """Effectue une prédiction avec un modèle spécifique."""
    try:
        model_id = request.args.get("model_id")
        if model_id not in available_models:
            return jsonify({"error": "Modèle non disponible"}), 400

        input_data = preprocess_input(request.args)
        if isinstance(input_data, dict):  # En cas d'erreur dans le prétraitement
            return jsonify(input_data), 400

        model = joblib.load(available_models[model_id])
        probabilities = model.predict_proba(input_data)[0]
        not_survived_prob = probabilities[0] * 100
        survived_prob = probabilities[1] * 100

        prediction = "Survecu" if survived_prob > not_survived_prob else "Pas survecu"

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


@app.route("/consensus", methods=["GET"])
def consensus():
    """Génère une prédiction basée sur la moyenne pondérée des 4 modèles."""
    try:
        input_data = preprocess_input(request.args)
        if isinstance(input_data, dict):  # En cas d'erreur dans le prétraitement
            return jsonify(input_data), 400

        results = []
        total_weight = sum(model_weights.values())
        weighted_probs = {"not_survived": 0, "survived": 0}

        for model_id, model_path in available_models.items():
            model = joblib.load(model_path)
            probabilities = model.predict_proba(input_data)[0]
            not_survived_prob = probabilities[0] * 100
            survived_prob = probabilities[1] * 100

            weight = model_weights[model_id]
            weighted_probs["not_survived"] += not_survived_prob * weight
            weighted_probs["survived"] += survived_prob * weight

            results.append({
                "model_id": model_id,
                "prediction": "Survecu" if survived_prob > not_survived_prob else "Pas survecu",
                "probabilities": {
                    "not_survived": f"{not_survived_prob:.2f}%",
                    "survived": f"{survived_prob:.2f}%"
                },
                "weight": round(weight, 2)
            })

        # Moyenne pondérée des probabilités
        avg_not_survived = weighted_probs["not_survived"] / total_weight
        avg_survived = weighted_probs["survived"] / total_weight
        final_prediction = "Survecu" if avg_survived > avg_not_survived else "Pas survecu"

        # Mise à jour des poids (slashing mechanism)
        for model in results:
            model_id = model["model_id"]
            survived_prob = float(model["probabilities"]["survived"].strip('%'))
            error = abs(survived_prob - avg_survived)

            if error < 5:  # Petit écart → Augmentation légère
                model_weights[model_id] = min(1.0, model_weights[model_id] + 0.05)
            elif error > 15:  # Grand écart → Réduction du poids
                model_weights[model_id] = max(0.1, model_weights[model_id] * 0.8)


        return jsonify({
            "individual_predictions": results,
            "consensus": {
                "prediction": final_prediction,
                "probabilities": {
                    "not_survived": f"{avg_not_survived:.2f}%",
                    "survived": f"{avg_survived:.2f}%"
                }
            },
            "updated_weights": {mid: round(w, 2) for mid, w in model_weights.items()}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
