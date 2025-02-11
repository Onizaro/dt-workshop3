import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Chargement et prétraitement des données
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

# Création et entraînement du modèle SVM
model = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
model.fit(X_train, y_train)

# Enregistrement du modèle et des encodeurs
joblib.dump(model, "models/anton_model.pth")
joblib.dump(label_encoders, "models/label_encoders.pth")
print("Modèle SVM et encodeurs sauvegardés avec succès !")
