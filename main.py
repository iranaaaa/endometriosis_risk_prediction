import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.model import train_xgboost, evaluate_model, plot_confusion_matrix, plot_roc_curve, explain_model_shap
import matplotlib.pyplot as plt

def main():
    print("Welcome to Endometriosis Risk Prediction Core Model")
    
    # Path to your dataset (replace with actual path)
    dataset_path = r"C:\Users\hp\.gemini\antigravity\scratch\endometriosis_risk_prediction\dataset.csv"
    
    try:
        # Load the dataset
        print(f"Loading dataset from: {dataset_path}...")
        df = pd.read_csv(dataset_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        
        # Determine the target column based on the dataset (default to 'label')
        target_col = 'label'
        
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(df, target_col=target_col)
        
        print("Training XGBoost Model...")
        model = train_xgboost(X_train, y_train)
        
        print("\n--- Model Evaluation ---")
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
        print("\nGenerating visual evaluations...")
        # Save plots to disk
        fig_cm = plot_confusion_matrix(y_test, y_pred)
        fig_cm.savefig("confusion_matrix.png")
        print("Saved: confusion_matrix.png")
        
        fig_roc = plot_roc_curve(y_test, y_prob)
        fig_roc.savefig("roc_curve.png")
        print("Saved: roc_curve.png")
        
        print("\nGenerating SHAP Global Explainability Plot...")
        explainer, shap_values = explain_model_shap(model, X_train, X_test)
        import shap
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        print("Saved: shap_summary.png")
        
        print("\nTraining and evaluation finished successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{dataset_path}'.")
        print("Please place your dataset file in this folder and name it 'dataset.csv' (or update the path in this code).")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
