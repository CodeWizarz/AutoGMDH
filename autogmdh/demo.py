# autogmdh/demo.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from .core import AutoGMDHRegressor

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
COLUMN_NAMES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
SEED = 42


def load_housing():
    data = pd.read_csv(DATA_URL, sep=r"\s+", header=None, names=COLUMN_NAMES)
    X = data.drop("MEDV", axis=1).values.astype(np.float32)
    y = data["MEDV"].values.astype(np.float32)
    return X, y


def run_housing_demo():
    print("Loading UCI Housing dataset...")
    X, y = load_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    print("Training AutoGMDHRegressor (Advanced Two-Stage GMDH + NAS)...")
    model = AutoGMDHRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAdvanced Self-Organizing Hybrid Test Results:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²:  {r2_score(y_test, y_pred):.4f}")

    # Baselines using the same scalers as AutoGMDHRegressor
    x_scaler = model.x_scaler
    y_scaler = model.y_scaler

    X_train_scaled = x_scaler.transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()

    print("\nBaseline Comparisons:")

    # Ridge Regression baseline
    ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train_scaled)
    y_pred_lr_scaled = ridge.predict(X_test_scaled).flatten()
    y_pred_lr = y_scaler.inverse_transform(
        y_pred_lr_scaled.reshape(-1, 1)
    ).flatten()
    print(
        f"Ridge MSE: {mean_squared_error(y_test, y_pred_lr):.4f}, "
        f"R²: {r2_score(y_test, y_pred_lr):.4f}"
    )

    # Standalone NN baseline
    nn_baseline = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    nn_baseline.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True, verbose=0
    )
    nn_baseline.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=400,
        batch_size=16,
        callbacks=[es],
        verbose=0,
    )
    y_pred_nn_scaled = nn_baseline.predict(X_test_scaled).flatten()
    y_pred_nn = y_scaler.inverse_transform(
        y_pred_nn_scaled.reshape(-1, 1)
    ).flatten()
    print(
        f"Standalone NN MSE: {mean_squared_error(y_test, y_pred_nn):.4f}, "
        f"R²: {r2_score(y_test, y_pred_nn):.4f}"
    )

    print("\nDemo complete.")


if __name__ == "__main__":
    run_housing_demo()
