import requests
import numpy as np

# URLs des différents modèles exposés via ngrok
urls = {
    "1": "http://127.0.0.1:5000/predict",
    "2": "http://<your_ngrok_url_2>/predict",
    "3": "http://<your_ngrok_url_3>/predict",
    "4": "http://127.0.0.1:4040/predict"
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

# Liste pour stocker les prédictions
predictions = []

# Envoyer une requête pour chaque modèle et récupérer les prédictions
for model_id, url in urls.items():
    params["model_id"] = model_id
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "prediction" in data:
            print(f"Modèle {model_id} - Prédiction: {data['prediction']}")
            predictions.append(data['prediction'])
        elif "probabilities" in data:
            print(f"Modèle {model_id} - Probabilités: {data['probabilities']}")
            predictions.append(data['probabilities'])
        else:
            print(f"Modèle {model_id} - Erreur: {data.get('error', 'No prediction result')}")
    except Exception as e:
        print(f"Erreur avec le modèle {model_id}: {str(e)}")

# Calculer la moyenne des prédictions
# Si vous avez des probabilités, vous pouvez extraire les pourcentages de survie.
# Par exemple :
numeric_predictions = []
for prediction in predictions:
    if isinstance(prediction, dict):  # Cas où on reçoit des probabilités
        if "survived" in prediction["probabilities"]:
            numeric_predictions.append(float(prediction["probabilities"]["survived"].strip('%')))
    else:  # Cas où on reçoit des prédictions de type "survécu" ou "pas survécu"
        numeric_predictions.append(1 if prediction == "Survécu" else 0)

# Calcul de la moyenne des prédictions (si 1 pour survécu, 0 pour pas survécu)
average_prediction = np.mean(numeric_predictions)
print(f"Moyenne des prédictions: {average_prediction * 100:.2f}%")
