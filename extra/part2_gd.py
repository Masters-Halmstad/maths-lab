import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for seaborn
sns.set_theme(style="whitegrid")

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

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    return (data - mean) / std, mean, std

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    beta = np.zeros(n)
    cost_history = []
    
    for _ in range(iterations):
        predictions = X @ beta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        beta = beta - learning_rate * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
        
    return beta, cost_history

def run_once(data):
    x = data[FEATURES].to_numpy()
    y = data["target"].to_numpy()

    indices = np.random.permutation(len(data))
    split = int(0.8 * len(data))

    train_x_raw = x[indices[:split]]
    test_x_raw = x[indices[split:]]
    train_y = y[indices[:split]]
    test_y = y[indices[split:]]
    
    # Normalize features
    train_x_norm, mean_x, std_x = normalize(train_x_raw)
    test_x_norm = (test_x_raw - mean_x) / std_x

    # Add intercept
    train_x = np.hstack([np.ones((train_x_norm.shape[0], 1)), train_x_norm])
    test_x = np.hstack([np.ones((test_x_norm.shape[0], 1)), test_x_norm])
    
    # Gradient Descent
    beta, cost_history = gradient_descent(train_x, train_y, learning_rate=0.1, iterations=2000)
    
    # Predictions
    y_pred = test_x @ beta
    # Errors
    errors = y_pred - test_y
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors**2))

    return beta, abs_errors, rmse, y_pred, test_y, cost_history


def main(data, runs=100):
    all_betas = []
    all_abs_errors = []
    all_rmses = []
    
    best_rmse = float('inf')
    best_model_data = None

    for _ in range(runs):
        beta, abs_errors, rmse, y_pred, test_y, cost_history = run_once(data)
        all_betas.append(beta)
        all_abs_errors.extend(abs_errors)
        all_rmses.append(rmse)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_data = {
                'beta': beta,
                'y_pred': y_pred,
                'test_y': test_y,
                'rmse': rmse,
                'cost_history': cost_history
            }

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
    
    # --- Plotting ---
    
    # 1. RMSE Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(all_rmses, kde=True, color='orange')
    plt.title('Distribution of RMSE over 100 Runs (Gradient Descent)')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.savefig('gd_rmse_histogram.png')
    plt.close()
    
    # 2. Coefficients with Error Bars
    feature_names = ['Intercept'] + FEATURES
    plt.figure(figsize=(12, 6))
    plt.errorbar(feature_names, mean_beta, yerr=std_beta, fmt='o', capsize=5, color='darkorange')
    plt.title('Mean Coefficients with Std Dev (Gradient Descent)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value (Normalized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('gd_coefficients.png')
    plt.close()
    
    # 3. Best Model Fit
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=best_model_data['test_y'], y=best_model_data['y_pred'], alpha=0.6, color='orange')
    
    # Ideal line
    min_val = min(best_model_data['test_y'].min(), best_model_data['y_pred'].min())
    max_val = max(best_model_data['test_y'].max(), best_model_data['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
    plt.title(f'Best Model (GD): Predicted vs Actual (RMSE: {best_rmse:.4f})')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.savefig('gd_best_model_fit.png')
    plt.close()
    
    # 4. Cost History (Convergence)
    plt.figure(figsize=(10, 6))
    plt.plot(best_model_data['cost_history'])
    plt.title('Cost Function Convergence (Best Run)')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.savefig('gd_convergence.png')
    plt.close()

    return {
        "mean_beta": mean_beta,
        "std_beta": std_beta,
        "mean_error": mean_error,
        "std_error": std_error,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "95pi": (lower_95, upper_95),
        "best_rmse": best_rmse
    }


if __name__ == "__main__":
    if not os.path.exists("Problem2_Dataset.csv"):
        print("Error: Problem2_Dataset.csv not found.")
    else:
        main_data = pd.read_csv("Problem2_Dataset.csv")
        stats = main(main_data, runs=100)

        print("-" * 50)
        print("Gradient Descent Results:")
        print("Average Coefficients (Beta):", stats["mean_beta"])
        print("Coefficients Std Dev:", stats["std_beta"])
        print("Mean Absolute Error:", stats["mean_error"])
        print("Std of Absolute Error:", stats["std_error"])
        print("RMSE Mean:", stats["rmse_mean"])
        print("RMSE Std Dev:", stats["rmse_std"])
        print("Best RMSE:", stats["best_rmse"])
        print("95% Prediction Interval of Errors:", stats["95pi"])
