import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/raw/supply_chain.csv")

# Select features
features = [
    "Price",
    "Availability",
    "Stock levels",
    "Lead times"
]

X = data[features]
y = data["Number of products sold"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = r2_score(y_test, y_pred)

print("Model R2 Score:", accuracy)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/demand_model.pkl")

print("Model trained and saved successfully.")