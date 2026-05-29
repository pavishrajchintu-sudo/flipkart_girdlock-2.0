import os
import joblib
import pandas as pd

from data_preprocessing import get_processed_data


def _normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def main():
    print("Loading and preprocessing test data...")
    _, X_test, index_col = get_processed_data("d:/gridlock_2.0/dataset/train.csv", "d:/gridlock_2.0/dataset/test.csv")

    print("Loading trained models...")
    lgb_model = joblib.load("models/lgbm_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    cat_model = None
    if os.path.exists("models/cat_model.pkl"):
        cat_model = joblib.load("models/cat_model.pkl")

    weight_path = "models/ensemble_weights.pkl"
    if os.path.exists(weight_path):
        payload = joblib.load(weight_path)
        if "weights" in payload:
            raw_weights = payload["weights"]
            lgb_weight = float(raw_weights.get("lgb", 0.75))
            xgb_weight = float(raw_weights.get("xgb", 0.25))
            cat_weight = float(raw_weights.get("cat", 0.0))
        else:
            lgb_weight = float(payload.get("lgb_weight", 0.75))
            xgb_weight = float(payload.get("xgb_weight", 0.25))
            cat_weight = 0.0
    else:
        lgb_weight = 0.75
        xgb_weight = 0.25
        cat_weight = 0.0
        print("ensemble_weights.pkl not found. Falling back to default 0.75/0.25.")

    weights = {"lgb": lgb_weight, "xgb": xgb_weight, "cat": cat_weight}
    if cat_model is None:
        weights["cat"] = 0.0
    weights = _normalize_weights(weights)

    print("Generating predictions...")
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    predictions = (lgb_preds * weights["lgb"]) + (xgb_preds * weights["xgb"])

    if cat_model is not None and weights["cat"] > 0:
        cat_preds = cat_model.predict(X_test)
        predictions = predictions + (cat_preds * weights["cat"])
        print(
            f"Using ensemble weights: {weights['lgb']:.2f} LGBM + "
            f"{weights['xgb']:.2f} XGB + {weights['cat']:.2f} CAT"
        )
    else:
        print(f"Using ensemble weights: {weights['lgb']:.2f} LGBM + {weights['xgb']:.2f} XGB")

    submission = pd.DataFrame({
        "Index": index_col if index_col is not None else range(len(predictions)),
        "demand": predictions,
    })
    submission["demand"] = submission["demand"].clip(lower=0)

    output_path = "d:/gridlock_2.0/submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission successfully saved to {output_path}")


if __name__ == "__main__":
    main()
