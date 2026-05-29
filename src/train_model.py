import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GroupKFold, KFold

from data_preprocessing import add_geohash_target_encodings, engineer_features, load_data


def calculate_score(actual, predicted):
    return max(0, 100 * metrics.r2_score(actual, predicted))


def _custom_r2_metric_lgb(y_true, y_pred):
    return "custom_r2", calculate_score(y_true, y_pred), True


def _get_splitter(groups, requested_folds, seed, split_mode):
    unique_groups = len(np.unique(groups))
    if split_mode == "group":
        n_splits = min(requested_folds, unique_groups)
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), n_splits
        return KFold(n_splits=3, shuffle=True, random_state=seed), 3
    if split_mode == "auto" and unique_groups >= requested_folds:
        return GroupKFold(n_splits=requested_folds), requested_folds
    return KFold(n_splits=3, shuffle=True, random_state=seed), 3


def _prepare_xy(frame):
    y = frame["demand"].astype(float).values
    x = frame.drop(columns=["demand", "Index"], errors="ignore")
    return x, y


def _extract_best_iteration(model, fallback):
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        try:
            best_iteration = model.get_booster().best_iteration
        except Exception:
            best_iteration = None
    if best_iteration is None:
        return fallback
    return max(1, int(best_iteration) + 1)


def _run_cv_for_model(model_name, base_frame, splits, params, seed):
    oof = np.zeros(len(base_frame), dtype=float)
    fold_scores = []
    best_iterations = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_fold = base_frame.iloc[train_idx].copy()
        val_fold = base_frame.iloc[val_idx].copy()

        train_encoded, val_encoded = add_geohash_target_encodings(train_fold, val_fold)
        x_train, y_train = _prepare_xy(train_encoded)
        x_val, y_val = _prepare_xy(val_encoded)

        if model_name == "lgb":
            model = lgb.LGBMRegressor(
                random_state=seed,
                n_jobs=-1,
                **params
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric=_custom_r2_metric_lgb,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=100)
                ],
            )
            preds = model.predict(x_val)
            best_iterations.append(_extract_best_iteration(model, params.get("n_estimators", 300)))
        else:
            model = xgb.XGBRegressor(
                random_state=seed,
                n_jobs=-1,
                objective="reg:squarederror",
                early_stopping_rounds=100,
                eval_metric="rmse",
                **params
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            preds = model.predict(x_val)
            best_iterations.append(_extract_best_iteration(model, params.get("n_estimators", 300)))

        oof[val_idx] = preds
        fold_score = calculate_score(y_val, preds)
        fold_scores.append(fold_score)
        print(f"{model_name.upper()} fold {fold_idx} score: {fold_score:.4f}")

    overall_score = calculate_score(base_frame["demand"].values, oof)
    print(f"{model_name.upper()} OOF score: {overall_score:.4f}")
    return {
        "oof": oof,
        "score": overall_score,
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "params": params,
    }


def _choose_best_run(model_name, base_frame, splits, grid, seed, budget_seconds, start_time):
    best_run = None
    for idx, params in enumerate(grid, start=1):
        if time.time() - start_time > budget_seconds:
            print(f"Stopping {model_name} search due to time budget.")
            break
        print(f"\n{model_name.upper()} config {idx}/{len(grid)}: {params}")
        run = _run_cv_for_model(model_name, base_frame, splits, params, seed)
        if best_run is None or run["score"] > best_run["score"]:
            best_run = run
    if best_run is None:
        raise RuntimeError(f"No {model_name} run finished within the runtime budget.")
    return best_run


def _optimize_blend_weight(lgb_oof, xgb_oof, y_true):
    best_weight = 0.5
    best_score = -1.0
    for w in np.arange(0, 1.01, 0.01):
        blended = (w * lgb_oof) + ((1 - w) * xgb_oof)
        score = calculate_score(y_true, blended)
        if score > best_score:
            best_score = score
            best_weight = float(round(w, 2))
    return best_weight, best_score


def _fit_full_models(train_frame, lgb_run, xgb_run, seed):
    x_full, y_full = _prepare_xy(train_frame)

    lgb_params = dict(lgb_run["params"])
    xgb_params = dict(xgb_run["params"])

    lgb_params["n_estimators"] = int(np.median(lgb_run["best_iterations"]))
    xgb_params["n_estimators"] = int(np.median(xgb_run["best_iterations"]))

    lgb_model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, **lgb_params)
    lgb_model.fit(x_full, y_full)

    xgb_model = xgb.XGBRegressor(
        random_state=seed,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="rmse",
        **xgb_params
    )
    xgb_model.fit(x_full, y_full, verbose=False)
    return lgb_model, xgb_model, lgb_params, xgb_params


def _get_lgb_grid(fast_mode):
    grid = [
        {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "max_depth": 10,
            "num_leaves": 256,
            "min_child_samples": 50,
            "reg_lambda": 0.10,
            "reg_alpha": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },
        {
            "n_estimators": 1100,
            "learning_rate": 0.025,
            "max_depth": 12,
            "num_leaves": 320,
            "min_child_samples": 40,
            "reg_lambda": 0.20,
            "reg_alpha": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    ]
    return grid[:1] if fast_mode else grid


def _get_xgb_grid(fast_mode):
    grid = [
        {
            "n_estimators": 450,
            "learning_rate": 0.03,
            "max_depth": 7,
            "min_child_weight": 5,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },
        {
            "n_estimators": 600,
            "learning_rate": 0.025,
            "max_depth": 8,
            "min_child_weight": 4,
            "reg_lambda": 2.5,
            "reg_alpha": 0.15,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
        },
    ]
    return grid[:1] if fast_mode else grid


def main():
    parser = argparse.ArgumentParser(description="Train CV-based LGBM/XGB ensemble for demand prediction.")
    parser.add_argument("--train-path", default="d:/gridlock_2.0/dataset/train.csv")
    parser.add_argument("--test-path", default="d:/gridlock_2.0/dataset/test.csv")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--split-mode",
        choices=["auto", "random", "group"],
        default="random",
        help="CV split strategy: random KFold, group-by-day GroupKFold, or auto",
    )
    parser.add_argument("--max-runtime-min", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-mode", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    budget_seconds = args.max_runtime_min * 60
    os.makedirs("models", exist_ok=True)

    print("Loading raw training data...")
    train_raw = load_data(args.train_path)
    print("Engineering base features...")
    train_base, _, _ = engineer_features(train_raw, artifacts_dir="models", fit_label_encoders=True)
    train_base = train_base.dropna(subset=["demand"]).reset_index(drop=True)

    groups = train_base["day"].fillna(-1).astype(int).values if "day" in train_base.columns else np.zeros(len(train_base))
    splitter, actual_folds = _get_splitter(groups, args.cv_folds, args.seed, args.split_mode)
    if isinstance(splitter, GroupKFold):
        split_iter = list(splitter.split(train_base, groups=groups))
        effective_split_mode = "group"
    else:
        split_iter = list(splitter.split(train_base))
        effective_split_mode = "random"

    print(
        f"Rows: {len(train_base)}, folds: {actual_folds}, fast_mode: {args.fast_mode}, "
        f"split_mode: {effective_split_mode}"
    )

    lgb_grid = _get_lgb_grid(args.fast_mode)
    xgb_grid = _get_xgb_grid(args.fast_mode)

    lgb_best = _choose_best_run("lgb", train_base, split_iter, lgb_grid, args.seed, budget_seconds, start_time)
    xgb_best = _choose_best_run("xgb", train_base, split_iter, xgb_grid, args.seed, budget_seconds, start_time)

    y_true = train_base["demand"].values
    blend_weight, blend_score = _optimize_blend_weight(lgb_best["oof"], xgb_best["oof"], y_true)

    print(f"\nBest LGBM OOF Score: {lgb_best['score']:.4f}")
    print(f"Best XGBoost OOF Score: {xgb_best['score']:.4f}")
    print(f"Best Blend Weight (LGBM): {blend_weight:.2f}")
    print(f"Best Ensemble OOF Score: {blend_score:.4f}")

    print("\nFitting full-data models with selected params...")
    full_train_encoded, _ = add_geohash_target_encodings(train_base.copy(), train_base.copy())
    lgb_model, xgb_model, final_lgb_params, final_xgb_params = _fit_full_models(
        full_train_encoded, lgb_best, xgb_best, args.seed
    )

    joblib.dump(lgb_model, "models/lgbm_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(
        {
            "lgb_weight": blend_weight,
            "xgb_weight": round(1.0 - blend_weight, 2),
            "oof_score": blend_score,
        },
        "models/ensemble_weights.pkl",
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_folds": actual_folds,
        "seed": args.seed,
        "fast_mode": args.fast_mode,
        "split_mode": effective_split_mode,
        "lgb": {
            "oof_score": lgb_best["score"],
            "fold_scores": lgb_best["fold_scores"],
            "selected_params": final_lgb_params,
        },
        "xgb": {
            "oof_score": xgb_best["score"],
            "fold_scores": xgb_best["fold_scores"],
            "selected_params": final_xgb_params,
        },
        "ensemble": {
            "lgb_weight": blend_weight,
            "xgb_weight": round(1.0 - blend_weight, 2),
            "oof_score": blend_score,
        },
        "runtime_seconds": round(time.time() - start_time, 2),
    }

    with open("models/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved models/lgbm_model.pkl, models/xgb_model.pkl, models/ensemble_weights.pkl")
    print("Saved models/training_summary.json")
    print(f"Total runtime: {summary['runtime_seconds']}s")


if __name__ == "__main__":
    main()
