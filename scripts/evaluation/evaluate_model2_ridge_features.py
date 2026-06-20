import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET_COLUMNS = [
    "rating",
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]

# ============================================================
# Utilities
# ============================================================

def semester_to_number(value):
    text = str(value).strip()
    if not text:
        return np.nan

    # Original model2 parser style
    m = re.search(r"(20\d{2})\D*([12])", text)
    if m:
        year = int(m.group(1))
        term = int(m.group(2))
        return year * 2 + (term - 1)

    m = re.search(r"(^|\D)(\d{2})\D*([12])($|\D)", text)
    if m:
        year = 2000 + int(m.group(2))
        term = int(m.group(3))
        return year * 2 + (term - 1)

    lowered = text.lower()
    m = re.search(r"(20\d{2})", lowered)
    if m:
        year = int(m.group(1))
        if "spring" in lowered:
            return year * 2
        if "fall" in lowered or "autumn" in lowered:
            return year * 2 + 1

    # Korean semester fallback: 26년 1학기, 25년 2학기
    m = re.search(r"(\d{2}|20\d{2})년\s*([12])학기", text)
    if m:
        year = int(m.group(1))
        if year < 100:
            year += 2000
        term = int(m.group(2))
        return year * 2 + (term - 1)

    return np.nan


def number_to_semester_label(num):
    year = int(num // 2)
    term = int(num % 2) + 1
    return f"{year}-{term}"


def parse_targets(value):
    if value is None or str(value).strip() == "":
        return DEFAULT_TARGET_COLUMNS
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_list(value, cast=str):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def safe_pearson(true, pred):
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if len(true) < 2:
        return np.nan
    if np.std(true) < 1e-8 or np.std(pred) < 1e-8:
        return np.nan
    return pearsonr(true, pred).statistic


def evaluate_vector(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Pearson_r": np.nan}
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "Pearson_r": safe_pearson(y_true, y_pred),
    }


def resolve_path(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path("OptiLife") / p
    if alt.exists():
        return alt
    return p

# ============================================================
# Loading
# ============================================================

def infer_prediction_columns(frame, target_columns, use_scaled_predictions=True):
    pred_columns = []
    for target in target_columns:
        if target == "rating":
            candidate = "rating"
        else:
            raw_candidate = f"review_pred_{target}"
            scaled_candidate = f"review_pred_{target}_scaled"
            candidate = scaled_candidate if use_scaled_predictions and scaled_candidate in frame.columns else raw_candidate
        if candidate not in frame.columns:
            raise ValueError(f"Missing column: {candidate}\nAvailable columns: {list(frame.columns)}")
        pred_columns.append(candidate)
    return pred_columns


def load_review_predictions(path, target_columns, use_scaled_predictions=True):
    path = resolve_path(path)
    frame = pd.read_csv(path)
    required = ["course_key", "semester"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"review prediction file missing columns: {missing}")

    pred_columns = infer_prediction_columns(frame, target_columns, use_scaled_predictions)
    frame = frame.copy()
    frame["semester_num"] = frame["semester"].apply(semester_to_number)

    before = len(frame)
    frame = frame[frame["semester_num"].notna()].reset_index(drop=True)
    after_semester = len(frame)

    for col in pred_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=pred_columns).reset_index(drop=True)
    after_pred = len(frame)

    if frame.empty:
        raise ValueError("No usable review predictions after filtering.")

    print(f"review prediction path: {path}")
    print(f"filter summary: raw={before}, valid_semester={after_semester}, valid_predictions={after_pred}")
    print("prediction columns used:")
    for target, col in zip(target_columns, pred_columns):
        print(f"  {target}: {col}")
    return frame, pred_columns


def build_course_semester_table(review_df, target_columns, pred_columns):
    agg_dict = {pred_col: "mean" for pred_col in pred_columns}
    if "raw_review_text" in review_df.columns:
        agg_dict["raw_review_text"] = "count"
    if "course_name" in review_df.columns:
        agg_dict["course_name"] = "first"
    if "professor" in review_df.columns:
        agg_dict["professor"] = "first"

    table = review_df.groupby(["course_key", "semester_num"], as_index=False).agg(agg_dict)

    if "raw_review_text" in table.columns:
        table = table.rename(columns={"raw_review_text": "review_count"})
    else:
        count_df = review_df.groupby(["course_key", "semester_num"], as_index=False).size().rename(columns={"size": "review_count"})
        table = table.merge(count_df, on=["course_key", "semester_num"], how="left")

    rename_map = {pred_col: f"semester_avg_{target}" for target, pred_col in zip(target_columns, pred_columns)}
    table = table.rename(columns=rename_map)
    table["semester_label"] = table["semester_num"].apply(number_to_semester_label)
    return table

# ============================================================
# Sample construction matching original code
# ============================================================

def safe_std(series):
    if len(series) <= 1:
        return 0.0
    value = series.std(ddof=0)
    if pd.isna(value):
        return 0.0
    return float(value)


def build_flatten_summary_samples(course_semester_df, target_columns, k_history, min_gap=1, max_gap=None):
    rows = []
    for course_key, group in course_semester_df.groupby("course_key"):
        group = group.sort_values("semester_num").reset_index(drop=True)
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            cur = group.iloc[i]
            hist = group.iloc[:i]
            prev = hist.iloc[-1]

            gap = int(cur["semester_num"] - prev["semester_num"])
            if gap < min_gap:
                continue
            if max_gap is not None and gap > max_gap:
                continue

            row = {
                "course_key": course_key,
                "prev_semester_num": prev["semester_num"],
                "target_semester_num": cur["semester_num"],
                "prev_semester": prev["semester_label"],
                "target_semester": cur["semester_label"],
                "semester_gap": gap,
                "history_length": len(hist),
                "historical_review_count": float(hist["review_count"].sum()),
                "prev_review_count": float(prev["review_count"]),
                "target_review_count": float(cur["review_count"]),
            }
            if "course_name" in group.columns:
                row["course_name"] = cur.get("course_name", "")
            if "professor" in group.columns:
                row["professor"] = cur.get("professor", "")

            for slot in range(1, k_history + 1):
                hist_pos = len(hist) - slot
                row[f"recent_{slot}_available"] = 1.0 if hist_pos >= 0 else 0.0
                if hist_pos >= 0:
                    h = hist.iloc[hist_pos]
                    row[f"recent_{slot}_semester_gap"] = float(cur["semester_num"] - h["semester_num"])
                    row[f"recent_{slot}_review_count"] = float(h["review_count"])
                    for target in target_columns:
                        row[f"recent_{slot}_{target}"] = float(h[f"semester_avg_{target}"])
                else:
                    row[f"recent_{slot}_semester_gap"] = 0.0
                    row[f"recent_{slot}_review_count"] = 0.0
                    for target in target_columns:
                        row[f"recent_{slot}_{target}"] = 0.0

            for target in target_columns:
                col = f"semester_avg_{target}"
                hist_values = hist[col]
                prev_value = float(prev[col])
                hist_mean = float(hist_values.mean())
                row[f"hist_mean_{target}"] = hist_mean
                row[f"hist_std_{target}"] = safe_std(hist_values)
                row[f"hist_min_{target}"] = float(hist_values.min())
                row[f"hist_max_{target}"] = float(hist_values.max())
                row[f"trend_from_mean_{target}"] = prev_value - hist_mean
                row[f"trend_last_diff_{target}"] = prev_value - float(hist.iloc[-2][col]) if len(hist) >= 2 else 0.0
                row[f"pred_last_value_{target}"] = prev_value
                row[f"pred_historical_mean_{target}"] = hist_mean
                row[f"true_{target}"] = float(cur[col])

            rows.append(row)

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("No model2 samples. Need at least two valid semesters per course.")

    numeric_cols = samples.select_dtypes(include=[np.number]).columns
    samples[numeric_cols] = samples[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return samples


def get_original_feature_columns(target_columns, k_history, use_last_diff_trend=True):
    feature_columns = [
        "semester_gap",
        "history_length",
        "historical_review_count",
        "prev_review_count",
    ]
    for slot in range(1, k_history + 1):
        feature_columns.extend([
            f"recent_{slot}_available",
            f"recent_{slot}_semester_gap",
            f"recent_{slot}_review_count",
        ])
        for target in target_columns:
            feature_columns.append(f"recent_{slot}_{target}")

    for target in target_columns:
        feature_columns.extend([
            f"hist_mean_{target}",
            f"hist_std_{target}",
            f"hist_min_{target}",
            f"hist_max_{target}",
            f"trend_from_mean_{target}",
        ])
        if use_last_diff_trend:
            feature_columns.append(f"trend_last_diff_{target}")

    return feature_columns


def make_feature_columns(target_columns, k_history, feature_mode, use_last_diff_trend=True):
    if feature_mode == "original_full":
        return get_original_feature_columns(target_columns, k_history, use_last_diff_trend)

    feature_columns = []
    if "gap" in feature_mode:
        feature_columns.append("semester_gap")
    if "length" in feature_mode:
        feature_columns.append("history_length")
    if "count" in feature_mode:
        feature_columns.extend(["historical_review_count", "prev_review_count"])

    if "recent" in feature_mode:
        for slot in range(1, k_history + 1):
            feature_columns.extend([
                f"recent_{slot}_available",
                f"recent_{slot}_semester_gap",
                f"recent_{slot}_review_count",
            ])
            for target in target_columns:
                feature_columns.append(f"recent_{slot}_{target}")

    for target in target_columns:
        if "hist_mean" in feature_mode:
            feature_columns.append(f"hist_mean_{target}")
        if "hist_stats" in feature_mode:
            feature_columns.extend([f"hist_std_{target}", f"hist_min_{target}", f"hist_max_{target}"])
        if "trend" in feature_mode:
            feature_columns.append(f"trend_from_mean_{target}")
            if use_last_diff_trend:
                feature_columns.append(f"trend_last_diff_{target}")

    seen = set()
    out = []
    for c in feature_columns:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# ============================================================
# Fitting/evaluation
# ============================================================

def split_samples(samples, val_size, seed, group_split=True):
    if group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(samples, groups=samples["course_key"]))
        return samples.iloc[train_idx].reset_index(drop=True), samples.iloc[val_idx].reset_index(drop=True)
    train_df, val_df = train_test_split(samples, test_size=val_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def fit_predict_ridge(train_df, val_df, feature_columns, y_cols, alpha, impute=True):
    X_train = train_df.reindex(columns=feature_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    X_val = val_df.reindex(columns=feature_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    y_train = train_df.reindex(columns=y_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if impute:
        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    else:
        X_train = X_train.fillna(0.0)
        X_val = X_val.fillna(0.0)
        steps = [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    model = Pipeline(steps)
    model.fit(X_train.values, y_train.values)
    return np.clip(model.predict(X_val.values), 1.0, 5.0)


def add_result(rows, kind, k, alpha, feature_mode, target_mode, target, y_true, y_pred, n_features, sort_metric):
    metric = evaluate_vector(y_true, y_pred)
    row = {
        "kind": kind,
        "k_history": k,
        "alpha": alpha,
        "feature_mode": feature_mode,
        "target_mode": target_mode,
        "target": target,
        "n_features": n_features,
        **metric,
    }
    # Higher is better for all leaderboard scores
    if sort_metric == "pearson":
        row["score"] = row["Pearson_r"]
    elif sort_metric == "mae":
        row["score"] = -row["MAE"]
    elif sort_metric == "rmse":
        row["score"] = -row["RMSE"]
    elif sort_metric == "mae_then_pearson":
        row["score"] = -row["MAE"]
    else:
        raise ValueError(f"Unknown sort_metric: {sort_metric}")
    rows.append(row)


def run_config_search(course_semester, target_columns, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_list(args.k_values, int)
    alpha_values = parse_list(args.alpha_values, float)
    feature_modes = parse_list(args.feature_modes, str)

    rows = []
    for k in k_values:
        print("\n" + "=" * 80)
        print(f"Building samples for k_history={k}")
        print("=" * 80)
        samples = build_flatten_summary_samples(
            course_semester,
            target_columns,
            k_history=k,
            min_gap=args.min_gap,
            max_gap=None if args.max_gap == 0 else args.max_gap,
        )
        train_df, val_df = split_samples(samples, args.val_size, args.seed, args.group_split)
        print(f"samples={len(samples)}, train={len(train_df)}, val={len(val_df)}, courses={samples['course_key'].nunique()}")

        y_cols_multi = [f"true_{target}" for target in target_columns]
        y_col_rating = ["true_rating"]
        y_val_multi = val_df[y_cols_multi].values
        y_val_rating = val_df["true_rating"].values
        pred_last = val_df[[f"pred_last_value_{target}" for target in target_columns]].values
        pred_hist = val_df[[f"pred_historical_mean_{target}" for target in target_columns]].values

        for baseline_name, pred in [("last_semester", pred_last), ("historical_mean", pred_hist)]:
            for i, target in enumerate(target_columns):
                add_result(rows, "baseline", k, np.nan, baseline_name, "none", target, y_val_multi[:, i], pred[:, i], 0, args.sort_metric)
            add_result(rows, "baseline", k, np.nan, baseline_name, "none", "average", y_val_multi.reshape(-1), pred.reshape(-1), 0, args.sort_metric)

        for feature_mode in feature_modes:
            feature_columns = make_feature_columns(target_columns, k, feature_mode, args.use_last_diff_trend)
            if not feature_columns:
                continue

            for alpha in alpha_values:
                if args.run_multi:
                    pred_multi = fit_predict_ridge(train_df, val_df, feature_columns, y_cols_multi, alpha, impute=args.use_imputer)
                    for i, target in enumerate(target_columns):
                        add_result(rows, "ridge", k, alpha, feature_mode, "multi", target, y_val_multi[:, i], pred_multi[:, i], len(feature_columns), args.sort_metric)
                    add_result(rows, "ridge", k, alpha, feature_mode, "multi", "average", y_val_multi.reshape(-1), pred_multi.reshape(-1), len(feature_columns), args.sort_metric)

                if args.run_rating_only:
                    pred_rating = fit_predict_ridge(train_df, val_df, feature_columns, y_col_rating, alpha, impute=args.use_imputer)
                    pred_rating = np.asarray(pred_rating)
                    if pred_rating.ndim == 2:
                        pred_rating = pred_rating[:, 0]
                    add_result(rows, "ridge", k, alpha, feature_mode, "rating_only", "rating", y_val_rating, pred_rating, len(feature_columns), args.sort_metric)

        samples.to_csv(output_dir / f"ridge_search_samples_k{k}.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "ridge_config_search_metrics.csv", index=False, encoding="utf-8-sig")

    rating = metrics[metrics["target"] == "rating"].copy()
    if args.sort_metric == "mae_then_pearson":
        rating = rating.sort_values(["MAE", "Pearson_r"], ascending=[True, False]).reset_index(drop=True)
    elif args.sort_metric == "mae":
        rating = rating.sort_values(["MAE", "Pearson_r"], ascending=[True, False]).reset_index(drop=True)
    elif args.sort_metric == "rmse":
        rating = rating.sort_values(["RMSE", "Pearson_r"], ascending=[True, False]).reset_index(drop=True)
    else:
        rating = rating.sort_values(["Pearson_r", "MAE"], ascending=[False, True]).reset_index(drop=True)
    rating.to_csv(output_dir / "ridge_rating_leaderboard.csv", index=False, encoding="utf-8-sig")

    average = metrics[metrics["target"] == "average"].copy()
    average = average.sort_values(["Pearson_r", "MAE"], ascending=[False, True]).reset_index(drop=True)
    average.to_csv(output_dir / "ridge_average_leaderboard.csv", index=False, encoding="utf-8-sig")

    print("\n=== Rating leaderboard top 30 ===")
    print(rating.head(30).to_string(index=False))

    print("\n=== Original-like config rows ===")
    mask = (
        (metrics["kind"] == "ridge")
        & (metrics["feature_mode"] == "original_full")
        & (metrics["alpha"] == 1.0)
        & (metrics["target"] == "rating")
    )
    print(metrics[mask].sort_values(["k_history", "target_mode"]).to_string(index=False))

    print("\nsaved:")
    print(output_dir / "ridge_config_search_metrics.csv")
    print(output_dir / "ridge_rating_leaderboard.csv")
    print(output_dir / "ridge_average_leaderboard.csv")

    return metrics, rating

# ============================================================
# CLI
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-predictions", default="OptiLife/scripts/examples/train_course_attribute_model/output/bert_course_attribute_model/bert_all_review_predictions.csv")
    parser.add_argument("--targets", default="rating,workload_label,teamwork_load_label,grading_strictness_label")
    parser.add_argument("--output-dir", default="OptiLife/outputs_model2_ridge_config_search_match_original")
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--group-split", action="store_true", default=True)
    parser.add_argument("--random-split", dest="group_split", action="store_false")
    parser.add_argument("--use-scaled-predictions", action="store_true", default=True)
    parser.add_argument("--use-raw-predictions", dest="use_scaled_predictions", action="store_false")
    parser.add_argument("--no-last-diff-trend", dest="use_last_diff_trend", action="store_false")
    parser.set_defaults(use_last_diff_trend=True)

    # This includes the exact original Ridge k=1/3/5 configuration.
    parser.add_argument("--k-values", default="1,2,3,4,5")
    parser.add_argument("--alpha-values", default="0.0001,0.001,0.01,0.1,0.3,1,3,10,30,100,300,1000")
    parser.add_argument(
        "--feature-modes",
        default=(
            "original_full,"
            "recent+hist_mean+gap+length+count,"
            "recent+hist_mean+hist_stats+gap+length+count,"
            "recent+hist_mean+trend+gap+length+count,"
            "recent+hist_mean+hist_stats+trend+gap+length+count,"
            "hist_mean+gap+length+count,"
            "hist_mean+hist_stats+gap+length+count,"
            "hist_mean+trend+gap+length+count,"
            "recent+gap+length+count,"
            "recent+hist_mean,"
            "hist_mean"
        ),
    )
    parser.add_argument("--sort-metric", default="mae_then_pearson", choices=["mae_then_pearson", "mae", "rmse", "pearson"])
    parser.add_argument("--run-multi", action="store_true", default=True)
    parser.add_argument("--no-multi", dest="run_multi", action="store_false")
    parser.add_argument("--run-rating-only", action="store_true", default=True)
    parser.add_argument("--no-rating-only", dest="run_rating_only", action="store_false")
    parser.add_argument("--use-imputer", action="store_true", default=True)
    parser.add_argument("--no-imputer", dest="use_imputer", action="store_false")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    target_columns = parse_targets(args.targets)
    review_df, pred_columns = load_review_predictions(args.review_predictions, target_columns, args.use_scaled_predictions)
    print("loaded review predictions:", len(review_df))
    print("course groups:", review_df["course_key"].nunique())
    print("targets:", target_columns)
    course_semester = build_course_semester_table(review_df, target_columns, pred_columns)
    print("course-semester rows:", len(course_semester))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    course_semester.to_csv(output_dir / "model2_course_semester_scores.csv", index=False, encoding="utf-8-sig")
    run_config_search(course_semester, target_columns, args)


if __name__ == "__main__":
    main()
