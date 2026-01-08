import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from part2 import run_once, load_data, FEATURES

# Set style for seaborn
sns.set_theme(style="whitegrid")

def main(data, runs=100):
    all_betas = []
    all_abs_errors = []
    all_rmses = []
    
    best_rmse = float('inf')
    best_model_data = None

    for _ in range(runs):
        beta, abs_errors, rmse, y_pred, test_y = run_once(data)
        all_betas.append(beta)
        all_abs_errors.extend(abs_errors)
        all_rmses.append(rmse)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_data = {
                'beta': beta,
                'y_pred': y_pred,
                'test_y': test_y,
                'rmse': rmse
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
    
    # --- Plotting ---
    
    # 1. RMSE Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(all_rmses, kde=True, color='skyblue')
    plt.title('Distribution of RMSE over 100 Runs (Normal Equation)')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.savefig('ne_rmse_histogram.png')
    plt.close()
    
    # 3. Best Model Fit
    plt.figure(figsize=(10, 6))
    if best_model_data:
        sns.scatterplot(x=best_model_data['test_y'], y=best_model_data['y_pred'], alpha=0.6)
        
        # Ideal line
        min_val = min(best_model_data['test_y'].min(), best_model_data['y_pred'].min())
        max_val = max(best_model_data['test_y'].max(), best_model_data['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
        
        plt.title(f'Best Model: Predicted vs Actual (RMSE: {best_rmse:.4f})')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.savefig('ne_best_model_fit.png')
        plt.close()

    return {
        "mean_beta": mean_beta,
        "std_beta": std_beta,
        "mean_error": mean_error,
        "std_error": std_error,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "best_rmse": best_rmse
    }


if __name__ == "__main__":
    main_data = load_data()
    if main_data is not None:
        stats = main(main_data, runs=10000)

        print("-" * 50)
        print("Normal Equation Results (100 runs):")
        print("Average Coefficients (Beta):", stats["mean_beta"])
        print("Coefficients Std Dev:", stats["std_beta"])
        print("Mean Absolute Error:", stats["mean_error"])
        print("Std of Absolute Error:", stats["std_error"])
        print("RMSE Mean:", stats["rmse_mean"])
        print("RMSE Std Dev:", stats["rmse_std"])
        print("Best RMSE:", stats["best_rmse"])
