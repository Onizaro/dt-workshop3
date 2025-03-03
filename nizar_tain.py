import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import LabelEncoder

# Chargement et prétraitement des données
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Enregistrement du modèle et des encodeurs
joblib.dump(model, "models/nizar_model.pth")
joblib.dump(label_encoders, "models/label_encoders.pth")
print("Modèle et encodeurs sauvegardés avec succès !")
