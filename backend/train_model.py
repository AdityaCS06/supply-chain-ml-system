import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

data = pd.read_csv("data/raw/supply_chain.csv")

# Select only important features
features = [
    "Price",
    "Availability",
    "Stock levels",
    "Lead times"
]

X = data[features]
y = data["Number of products sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/demand_model.pkl")

print("Model trained successfully")