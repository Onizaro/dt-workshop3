from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
print("Modèle et encodeurs sauvegardés avec succès !")
