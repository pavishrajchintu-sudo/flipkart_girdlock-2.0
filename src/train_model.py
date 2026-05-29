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

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


def calculate_score(actual, predicted):
    return max(0, 100 * metrics.r2_score(actual, predicted))


def _custom_r2_metric_lgb(y_true, y_pred):
    return "custom_r2", calculate_score(y_true, y_pred), True


def _log_event(event_name, **kwargs):
    payload = {
        "event": event_name,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=True))


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
    fold_durations = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_start = time.time()
        train_fold = base_frame.iloc[train_idx].copy()
        val_fold = base_frame.iloc[val_idx].copy()

        train_encoded, val_encoded = add_geohash_target_encodings(train_fold, val_fold)
        x_train, y_train = _prepare_xy(train_encoded)
        x_val, y_val = _prepare_xy(val_encoded)

        if model_name == "lgb":
            model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, **params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric=_custom_r2_metric_lgb,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            preds = model.predict(x_val)
            best_iterations.append(_extract_best_iteration(model, params.get("n_estimators", 300)))
        elif model_name == "xgb":
            model = xgb.XGBRegressor(
                random_state=seed,
                n_jobs=-1,
                objective="reg:squarederror",
                early_stopping_rounds=100,
                eval_metric="rmse",
                **params,
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            preds = model.predict(x_val)
            best_iterations.append(_extract_best_iteration(model, params.get("n_estimators", 300)))
        else:
            model = cb.CatBoostRegressor(
                random_seed=seed,
                loss_function="RMSE",
                eval_metric="RMSE",
                verbose=False,
                **params,
            )
            model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, early_stopping_rounds=100)
            preds = model.predict(x_val)
            best_iterations.append(_extract_best_iteration(model, params.get("iterations", 300)))

        oof[val_idx] = preds
        fold_score = calculate_score(y_val, preds)
        fold_scores.append(fold_score)
        fold_duration = round(time.time() - fold_start, 2)
        fold_durations.append(fold_duration)
        _log_event(
            "fold_complete",
            model=model_name,
            fold=fold_idx,
            score=round(fold_score, 4),
            duration_s=fold_duration,
            val_rows=len(val_idx),
        )

    overall_score = calculate_score(base_frame["demand"].values, oof)
    _log_event(
        "model_cv_complete",
        model=model_name,
        oof_score=round(overall_score, 4),
        mean_fold_duration_s=round(float(np.mean(fold_durations)), 2),
    )
    return {
        "oof": oof,
        "score": overall_score,
        "fold_scores": fold_scores,
        "fold_durations": fold_durations,
        "best_iterations": best_iterations,
        "params": params,
    }


def _choose_best_run(model_name, base_frame, splits, grid, seed, budget_seconds, start_time):
    best_run = None
    for idx, params in enumerate(grid, start=1):
        if time.time() - start_time > budget_seconds:
            _log_event("time_budget_stop", model=model_name, checked_configs=idx - 1)
            break
        _log_event("config_start", model=model_name, config_idx=idx, total_configs=len(grid), params=params)
        run = _run_cv_for_model(model_name, base_frame, splits, params, seed)
        if best_run is None or run["score"] > best_run["score"]:
            best_run = run
            _log_event("new_best", model=model_name, score=round(run["score"], 4), params=params)
    if best_run is None:
        raise RuntimeError(f"No {model_name} run finished within the runtime budget.")
    return best_run


def _optimize_blend_weights(oof_preds, y_true):
    names = list(oof_preds.keys())
    if len(names) == 2:
        a, b = names
        best_score = -1
        best_weights = {a: 0.5, b: 0.5}
        for w in np.arange(0, 1.01, 0.01):
            blended = (w * oof_preds[a]) + ((1 - w) * oof_preds[b])
            score = calculate_score(y_true, blended)
            if score > best_score:
                best_score = score
                best_weights = {a: round(float(w), 2), b: round(float(1 - w), 2)}
        return best_weights, best_score

    if len(names) == 3:
        a, b, c = names
        best_score = -1
        best_weights = {a: 0.34, b: 0.33, c: 0.33}
        for wa in np.arange(0, 1.01, 0.05):
            for wb in np.arange(0, 1.01 - wa, 0.05):
                wc = 1.0 - wa - wb
                if wc < 0:
                    continue
                blended = (wa * oof_preds[a]) + (wb * oof_preds[b]) + (wc * oof_preds[c])
                score = calculate_score(y_true, blended)
                if score > best_score:
                    best_score = score
                    best_weights = {
                        a: round(float(wa), 2),
                        b: round(float(wb), 2),
                        c: round(float(wc), 2),
                    }
        norm = sum(best_weights.values())
        if norm > 0:
            best_weights = {k: round(v / norm, 2) for k, v in best_weights.items()}
        return best_weights, best_score

    raise ValueError("Blend optimization expects 2 or 3 models.")


def _fit_full_models(train_frame, lgb_run, xgb_run, seed, cat_run=None):
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
        **xgb_params,
    )
    xgb_model.fit(x_full, y_full, verbose=False)

    cat_model = None
    cat_params = None
    if cat_run is not None:
        cat_params = dict(cat_run["params"])
        cat_params["iterations"] = int(np.median(cat_run["best_iterations"]))
        cat_model = cb.CatBoostRegressor(
            random_seed=seed,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=False,
            **cat_params,
        )
        cat_model.fit(x_full, y_full)

    return lgb_model, xgb_model, cat_model, lgb_params, xgb_params, cat_params


def _get_lgb_grid(fast_mode):
    grid = [
        {
            "n_estimators": 1400,
            "learning_rate": 0.02,
            "max_depth": 10,
            "num_leaves": 320,
            "min_child_samples": 40,
            "reg_lambda": 0.2,
            "reg_alpha": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 1800,
            "learning_rate": 0.018,
            "max_depth": 12,
            "num_leaves": 384,
            "min_child_samples": 30,
            "reg_lambda": 0.3,
            "reg_alpha": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
        },
    ]
    return grid[:1] if fast_mode else grid


def _get_xgb_grid(fast_mode):
    grid = [
        {
            "n_estimators": 900,
            "learning_rate": 0.02,
            "max_depth": 8,
            "min_child_weight": 4,
            "reg_lambda": 2.5,
            "reg_alpha": 0.12,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 1200,
            "learning_rate": 0.015,
            "max_depth": 9,
            "min_child_weight": 3,
            "reg_lambda": 3.0,
            "reg_alpha": 0.18,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]
    return grid[:1] if fast_mode else grid


def _get_cat_grid(fast_mode):
    grid = [
        {
            "iterations": 700,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "subsample": 0.8,
        },
        {
            "iterations": 1000,
            "learning_rate": 0.02,
            "depth": 9,
            "l2_leaf_reg": 6.0,
            "subsample": 0.85,
        },
    ]
    return grid[:1] if fast_mode else grid


def main():
    parser = argparse.ArgumentParser(description="Train CV-based LGBM/XGB/CatBoost ensemble for demand prediction.")
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
    parser.add_argument("--use-catboost", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    budget_seconds = args.max_runtime_min * 60
    os.makedirs("models", exist_ok=True)

    _log_event("run_start", seed=args.seed, max_runtime_min=args.max_runtime_min, split_mode=args.split_mode)

    train_raw = load_data(args.train_path)
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

    _log_event("data_ready", rows=len(train_base), folds=actual_folds, split_mode=effective_split_mode)

    lgb_best = _choose_best_run("lgb", train_base, split_iter, _get_lgb_grid(args.fast_mode), args.seed, budget_seconds, start_time)
    xgb_best = _choose_best_run("xgb", train_base, split_iter, _get_xgb_grid(args.fast_mode), args.seed, budget_seconds, start_time)

    cat_best = None
    if args.use_catboost:
        if CATBOOST_AVAILABLE:
            cat_best = _choose_best_run(
                "cat",
                train_base,
                split_iter,
                _get_cat_grid(args.fast_mode),
                args.seed,
                budget_seconds,
                start_time,
            )
        else:
            _log_event("catboost_unavailable", note="Package not installed; continuing with LGB+XGB only")

    y_true = train_base["demand"].values
    oof_preds = {"lgb": lgb_best["oof"], "xgb": xgb_best["oof"]}
    if cat_best is not None:
        oof_preds["cat"] = cat_best["oof"]
    blend_weights, blend_score = _optimize_blend_weights(oof_preds, y_true)

    _log_event("best_scores", lgb=round(lgb_best["score"], 4), xgb=round(xgb_best["score"], 4), cat=None if cat_best is None else round(cat_best["score"], 4))
    _log_event("blend_selected", weights=blend_weights, score=round(blend_score, 4))

    full_train_encoded, _ = add_geohash_target_encodings(train_base.copy(), train_base.copy())
    lgb_model, xgb_model, cat_model, final_lgb_params, final_xgb_params, final_cat_params = _fit_full_models(
        full_train_encoded,
        lgb_best,
        xgb_best,
        args.seed,
        cat_run=cat_best,
    )

    joblib.dump(lgb_model, "models/lgbm_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    if cat_model is not None:
        joblib.dump(cat_model, "models/cat_model.pkl")
    joblib.dump({"weights": blend_weights, "oof_score": blend_score}, "models/ensemble_weights.pkl")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_folds": actual_folds,
        "seed": args.seed,
        "fast_mode": args.fast_mode,
        "split_mode": effective_split_mode,
        "use_catboost": args.use_catboost,
        "catboost_available": CATBOOST_AVAILABLE,
        "lgb": {
            "oof_score": lgb_best["score"],
            "fold_scores": lgb_best["fold_scores"],
            "fold_durations": lgb_best["fold_durations"],
            "selected_params": final_lgb_params,
        },
        "xgb": {
            "oof_score": xgb_best["score"],
            "fold_scores": xgb_best["fold_scores"],
            "fold_durations": xgb_best["fold_durations"],
            "selected_params": final_xgb_params,
        },
        "ensemble": {"weights": blend_weights, "oof_score": blend_score},
        "runtime_seconds": round(time.time() - start_time, 2),
    }
    if cat_best is not None:
        summary["cat"] = {
            "oof_score": cat_best["score"],
            "fold_scores": cat_best["fold_scores"],
            "fold_durations": cat_best["fold_durations"],
            "selected_params": final_cat_params,
        }

    with open("models/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _log_event("run_complete", runtime_seconds=summary["runtime_seconds"], summary_path="models/training_summary.json")


if __name__ == "__main__":
    main()
