import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import joblib

# Charger le dataset Titanic
def load_and_prepare_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    data = data[["Pclass", "Age", "SibSp", "Fare", "Survived"]].dropna()
    X = data[["Pclass", "Age", "SibSp", "Fare"]]
    y = data["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Encodage des variables catégoriques
label_encoders = {}
for col in ["sex", "class", "embarked"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Entraîner le modèle
X_train, X_test, y_train, y_test = load_and_prepare_data()
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Évaluer la performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Enregistrement du modèle et des encodeurs
joblib.dump(model, "models/anton_model.pth")
joblib.dump(label_encoders, "models/label_encoders.pth")
print("Modèle et encodeurs sauvegardés avec succès !")

# Créer l'API Flask
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Récupérer les arguments depuis l'URL
    try:
        pclass = float(request.args.get('Pclass', 0))
        age = float(request.args.get('Age', 0))
        sibsp = float(request.args.get('SibSp', 0))
        fare = float(request.args.get('Fare', 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input parameters"}), 400

    # Faire une prédiction
    input_data = [[pclass, age, sibsp, fare]]
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
