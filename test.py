import requests
import json

API_URL = "http://127.0.0.1:5000/consensus"

BALANCE_FILE = "model_balances.json"

try:
    with open(BALANCE_FILE, "r") as f:
        model_balances = json.load(f)
except FileNotFoundError:
    model_balances = {"1": 1000, "2": 1000, "3": 1000, "4": 1000}
    with open(BALANCE_FILE, "w") as f:
        json.dump(model_balances, f)

# 30 jeux de données différents
test_prompts = [
    {"sex": "male", "age": "22", "fare": "7.25", "class": "Third", "sibsp": "1", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "38", "fare": "71.28", "class": "First", "sibsp": "1", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "male", "age": "35", "fare": "8.05", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "27", "fare": "12.47", "class": "Second", "sibsp": "0", "parch": "0", "embarked": "Queenstown"},
    {"sex": "male", "age": "45", "fare": "35.50", "class": "First", "sibsp": "0", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "18", "fare": "9.00", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "male", "age": "50", "fare": "26.00", "class": "Second", "sibsp": "1", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "30", "fare": "40.00", "class": "First", "sibsp": "1", "parch": "1", "embarked": "Cherbourg"},
    {"sex": "male", "age": "28", "fare": "15.50", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Queenstown"},
    {"sex": "female", "age": "22", "fare": "80.00", "class": "First", "sibsp": "0", "parch": "1", "embarked": "Southampton"},
    {"sex": "male", "age": "60", "fare": "50.00", "class": "First", "sibsp": "1", "parch": "1", "embarked": "Cherbourg"},
    {"sex": "female", "age": "10", "fare": "20.00", "class": "Third", "sibsp": "0", "parch": "2", "embarked": "Southampton"},
    {"sex": "male", "age": "25", "fare": "15.00", "class": "Second", "sibsp": "0", "parch": "1", "embarked": "Queenstown"},
    {"sex": "female", "age": "50", "fare": "75.00", "class": "First", "sibsp": "1", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "male", "age": "33", "fare": "10.50", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "40", "fare": "90.00", "class": "First", "sibsp": "1", "parch": "1", "embarked": "Southampton"},
    {"sex": "male", "age": "19", "fare": "8.00", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Queenstown"},
    {"sex": "female", "age": "29", "fare": "30.00", "class": "Second", "sibsp": "1", "parch": "1", "embarked": "Cherbourg"},
    {"sex": "male", "age": "45", "fare": "20.00", "class": "Third", "sibsp": "0", "parch": "1", "embarked": "Southampton"},
    {"sex": "female", "age": "32", "fare": "60.00", "class": "First", "sibsp": "0", "parch": "0", "embarked": "Southampton"},
    {"sex": "male", "age": "55", "fare": "12.00", "class": "Second", "sibsp": "1", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "female", "age": "16", "fare": "5.00", "class": "Third", "sibsp": "0", "parch": "2", "embarked": "Southampton"},
    {"sex": "male", "age": "37", "fare": "45.00", "class": "First", "sibsp": "0", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "female", "age": "26", "fare": "25.00", "class": "Second", "sibsp": "0", "parch": "0", "embarked": "Queenstown"},
    {"sex": "male", "age": "62", "fare": "80.00", "class": "First", "sibsp": "1", "parch": "1", "embarked": "Southampton"},
    {"sex": "female", "age": "21", "fare": "10.00", "class": "Third", "sibsp": "0", "parch": "0", "embarked": "Cherbourg"},
    {"sex": "male", "age": "41", "fare": "22.00", "class": "Second", "sibsp": "0", "parch": "1", "embarked": "Queenstown"},
    {"sex": "female", "age": "36", "fare": "70.00", "class": "First", "sibsp": "1", "parch": "0", "embarked": "Southampton"},
    {"sex": "male", "age": "48", "fare": "18.00", "class": "Third", "sibsp": "1", "parch": "0", "embarked": "Southampton"},
    {"sex": "female", "age": "14", "fare": "50.00", "class": "First", "sibsp": "0", "parch": "2", "embarked": "Cherbourg"}
]

# Envoi des requêtes et application du slashing
for i, params in enumerate(test_prompts, start=1):
    response = requests.get(API_URL, params=params)
    data = response.json()
    print(f"Test {i}:", data, "\n")
    
    # Mise à jour des soldes et des poids en fonction des performances des modèles
    if "individual_predictions" in data:
        avg_survived = float(data["consensus"]["probabilities"]["survived"].strip('%'))
        
        for model in data["individual_predictions"]:
            model_id = model["model_id"]
            survived_prob = float(model["probabilities"]["survived"].strip('%'))
            error = abs(survived_prob - avg_survived)

            # Mise à jour du solde
            if error > 15:  
                model_balances[model_id] -= 50
            elif error < 5:  
                model_balances[model_id] += 10

            # Mise à jour du poids (plus précis = plus de poids)
            model["weight"] = max(1, 100 - error)

    # Sauvegarde des soldes mis à jour
    with open(BALANCE_FILE, "w") as f:
        json.dump(model_balances, f)
