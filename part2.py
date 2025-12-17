import pandas as pd
import numpy as np
import os

FEATURES = [
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
    "LSTAT",
]


def load_data(filepath="Problem2_Dataset.csv"):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    return pd.read_csv(filepath)


def run_once(data):
    x = data[FEATURES].to_numpy()
    y = data["target"].to_numpy()

    indices = np.random.permutation(len(data))
    split = int(0.8 * len(data))

    train_x = x[indices[:split]]
    test_x = x[indices[split:]]
    train_y = y[indices[:split]]
    test_y = y[indices[split:]]

    # Add intercept/bais term
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])
    # Normal equation
    # beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(train_x.T @ train_x) @ (train_x.T @ train_y)
    # Predictions
    y_pred = test_x @ beta
    # Errors
    errors = y_pred - test_y
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors**2))

    return beta, abs_errors, rmse, y_pred, test_y


if __name__ == "__main__":
    main_data = load_data()
    if main_data is not None:
        beta, abs_errors, rmse, y_pred, test_y = run_once(main_data)

        print("-" * 50)
        print("Single Run Result (Normal Equation):")
        print("Coefficients (Beta):", beta)
        print("RMSE:", rmse)
        print("Mean Absolute Error:", np.mean(abs_errors))

