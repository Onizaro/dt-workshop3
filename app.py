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

@app.route("/consensus", methods=["GET"])
def consensus():
    """Génère une prédiction basée sur la moyenne des 4 modèles."""
    try:
        input_data = preprocess_input(request.args)
        if isinstance(input_data, dict):  # En cas d'erreur dans le prétraitement
            return jsonify(input_data), 400

        results = []
        total_probs = {"not_survived": 0, "survived": 0}

        for model_id, model_path in available_models.items():
            model = joblib.load(model_path)
            probabilities = model.predict_proba(input_data)[0]
            not_survived_prob = probabilities[0] * 100
            survived_prob = probabilities[1] * 100

            total_probs["not_survived"] += not_survived_prob
            total_probs["survived"] += survived_prob

            results.append({
                "model_id": model_id,
                "prediction": "Survecu" if survived_prob > not_survived_prob else "Pas survecu",
                "probabilities": {
                    "not_survived": f"{not_survived_prob:.2f}%",
                    "survived": f"{survived_prob:.2f}%"
                }
            })

        # Moyenne des probabilités
        avg_not_survived = total_probs["not_survived"] / len(available_models)
        avg_survived = total_probs["survived"] / len(available_models)
        final_prediction = "Survecu" if avg_survived > avg_not_survived else "Pas survecu"

        return jsonify({
            "individual_predictions": results,
            "consensus": {
                "prediction": final_prediction,
                "probabilities": {
                    "not_survived": f"{avg_not_survived:.2f}%",
                    "survived": f"{avg_survived:.2f}%"
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
