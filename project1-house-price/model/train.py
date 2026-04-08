import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../data/AmesHousing.csv")

# ----------------------------
# Select better features
# ----------------------------
FEATURES = [
    'Gr Liv Area',
    'Overall Qual',
    'Garage Cars',
    'Total Bsmt SF',
    'Full Bath',
    'Year Built',
    'Overall Cond'
]

TARGET = 'SalePrice'

df = df[FEATURES + [TARGET]]

# ----------------------------
# Handle missing values
# ----------------------------
df.fillna(df.mean(), inplace=True)

# ----------------------------
# Split data
# ----------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scale features (optional for RF but okay)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train model (UPGRADED)
# ----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ----------------------------
# Predictions
# ----------------------------
y_pred = model.predict(X_test_scaled)

# ----------------------------
# Evaluation
# ----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# ----------------------------
# Save model & scaler
# ----------------------------
with open("../app/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../app/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model trained & saved successfully")