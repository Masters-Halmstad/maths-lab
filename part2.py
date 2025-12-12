import pandas as pd
import numpy as np

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


def run_once(data):
    x = data[FEATURES].to_numpy()
    y = data["target"].to_numpy()

    indices = np.random.permutation(len(data))
    split = int(0.8 * len(data))

    train_x = x[indices[:split]]
    test_x = x[indices[split:]]
    train_y = y[indices[:split]]
    test_y = y[indices[split:]]

    # Add intercept
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])
    # Normal equation
    beta = np.linalg.inv(train_x.T @ train_x) @ (train_x.T @ train_y)
    # Predictions
    y_pred = test_x @ beta
    # Errors
    errors = y_pred - test_y
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors**2))

    return beta, abs_errors, rmse


def main(data, runs=100):
    all_betas = []
    all_abs_errors = []
    all_rmses = []

    for _ in range(runs):
        beta, abs_errors, rmse = run_once(data)
        all_betas.append(beta)
        all_abs_errors.extend(abs_errors)
        all_rmses.append(rmse)

    all_betas = np.array(all_betas)
    mean_beta = np.mean(all_betas, axis=0)
    std_beta = np.std(all_betas, axis=0)

    all_abs_errors = np.array(all_abs_errors)
    mean_error = np.mean(all_abs_errors)
    std_error = np.std(all_abs_errors)

    all_rmses = np.array(all_rmses)
    rmse_mean = np.mean(all_rmses)
    rmse_std = np.std(all_rmses)

    # 95% prediction interval
    lower_95 = np.percentile(all_abs_errors, 2.5)
    upper_95 = np.percentile(all_abs_errors, 97.5)

    return {
        "mean_beta": mean_beta,
        "std_beta": std_beta,
        "mean_error": mean_error,
        "std_error": std_error,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "95pi": (lower_95, upper_95),
    }


if __name__ == "__main__":
    main_data = pd.read_csv("Problem2_Dataset.csv")
    stats = main(main_data, runs=100)

    print("-" * 50)
    print("Average Coefficients (Beta):", stats["mean_beta"])
    print("Coefficients Std Dev:", stats["std_beta"])
    print("Mean Absolute Error:", stats["mean_error"])
    print("Std of Absolute Error:", stats["std_error"])
    print("RMSE Mean:", stats["rmse_mean"])
    print("RMSE Std Dev:", stats["rmse_std"])
    print("95% Prediction Interval of Errors:", stats["95pi"])
