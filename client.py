import requests
import numpy as np

# URLs des modèles exposés via ngrok
urls = {
    "1": "https://39a2-37-169-109-180.ngrok-free.app/predict",
    "2": "https://0610-89-30-29-68.ngrok-free.app/predict",
    "3": "http://07b0-89-30-29-68.ngrok-free.app/predict",
    "4": "https://ae28-89-30-29-68.ngrok-free.app/predict"
}

# Paramètres à passer à la requête
params = {
    "sex": "male",
    "age": 22,
    "fare": 7.25,
    "class": "Third",
    "sibsp": 1,
    "parch": 0,
    "embarked": "Southampton"
}

predictions = []
numeric_predictions = []
errors = []

# Envoyer une requête pour chaque modèle et récupérer les prédictions
for model_id, url in urls.items():
    try:
        response = requests.get(url, params=params, timeout=5)  # Timeout pour éviter de bloquer
        data = response.json()

        if "probabilities" in data:
            not_survived_prob = float(data["probabilities"]["not_survived"].strip('%'))
            survived_prob = float(data["probabilities"]["survived"].strip('%'))
            numeric_predictions.append(survived_prob)
            predictions.append({
                "model_id": model_id,
                "prediction": data["prediction"],
                "probabilities": data["probabilities"]
            })
            print(f"Modèle {model_id} - Prédiction: {data['prediction']} (Survécu: {survived_prob:.2f}%)")

        else:
            errors.append(f"Modèle {model_id} - Erreur: {data.get('error', 'No valid prediction')}")

    except Exception as e:
        errors.append(f"Modèle {model_id} - Erreur: {str(e)}")

# Affichage des erreurs
if errors:
    print("\nErreurs rencontrées :")
    for error in errors:
        print(error)

# Calcul de la moyenne des probabilités de survie
if numeric_predictions:
    average_survival = np.mean(numeric_predictions)
    final_prediction = "Survécu" if average_survival > 50 else "Pas survécu"

    print(f"\nMoyenne des probabilités de survie: {average_survival:.2f}%")
    print(f"Prédiction finale basée sur le consensus : {final_prediction}")
else:
    print("\nAucune prédiction valide reçue.")
