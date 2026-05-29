import pandas as pd
import joblib
import os
from data_preprocessing import get_processed_data

def main():
    print("Loading and preprocessing test data...")
    _, X_test, index_col = get_processed_data('d:/gridlock_2.0/dataset/train.csv', 'd:/gridlock_2.0/dataset/test.csv')
    print("Loading trained models...")
    lgb_model = joblib.load('models/lgbm_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    weight_path = 'models/ensemble_weights.pkl'
    if os.path.exists(weight_path):
        weights = joblib.load(weight_path)
        lgb_weight = float(weights.get('lgb_weight', 0.75))
        xgb_weight = float(weights.get('xgb_weight', 0.25))
    else:
        lgb_weight = 0.75
        xgb_weight = 0.25
        print("ensemble_weights.pkl not found. Falling back to default 0.75/0.25.")
    print("Generating predictions...")
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    print(f"Using ensemble weights: {lgb_weight:.2f} LGBM + {xgb_weight:.2f} XGB")
    predictions = (lgb_preds * lgb_weight) + (xgb_preds * xgb_weight)
    submission = pd.DataFrame({
        'Index': index_col if index_col is not None else range(len(predictions)),
        'demand': predictions
    })
    submission['demand'] = submission['demand'].clip(lower=0)
    output_path = 'd:/gridlock_2.0/submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"Submission successfully saved to {output_path}")

if __name__ == "__main__":
    main()
