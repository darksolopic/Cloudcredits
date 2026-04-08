import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

if os.path.exists("data/diabetes.csv"):
    df = pd.read_csv("data/diabetes.csv")
else:
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)


# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved")