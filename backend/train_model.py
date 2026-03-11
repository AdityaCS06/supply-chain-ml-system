import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATA_PATH = "data/raw/supply_chain.csv"
MODEL_PATH = "models/demand_model.pkl"


def load_data():
    """Load dataset"""
    data = pd.read_csv(DATA_PATH)
    return data


def preprocess_data(data):
    """Select features and target"""
    
    features = [
        "Price",
        "Availability",
        "Stock levels",
        "Lead times"
    ]

    X = data[features]
    y = data["Number of products sold"]

    return X, y


def train_model(X_train, y_train):
    """Train Random Forest model"""
    
    model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=18,
    random_state=162
        # n_estimators=100,
        # random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nModel Evaluation")
    print("----------------")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")


def save_model(model):
    """Save trained model"""
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved at:", MODEL_PATH)


def main():

    print("Loading dataset...")
    data = load_data()

    print("Preprocessing data...")
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()