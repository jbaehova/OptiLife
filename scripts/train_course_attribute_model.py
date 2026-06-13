import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupKFold


DEFAULT_TARGET_COLUMNS = [
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]


# ============================================================
# Common utilities
# ============================================================

def clean_text(x):
    return "" if pd.isna(x) else str(x).strip()


def make_course_key(df):
    return (
        df["course_name"].astype(str).str.strip()
        + "__"
        + df["professor"].fillna("").astype(str).str.strip()
    )


def parse_targets(value):
    if value is None or str(value).strip() == "":
        return DEFAULT_TARGET_COLUMNS
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_pearson(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)

    if len(true) < 2:
        return np.nan
    if np.std(true) < 1e-8 or np.std(pred) < 1e-8:
        return np.nan

    return pearsonr(true, pred).statistic


def evaluate_predictions(y_true, y_pred, label):
    return {
        "model": label,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "Pearson_r": safe_pearson(y_true, y_pred),
    }


def evaluate_group_predictions(y_true, y_pred, target_columns, label):
    rows = []

    for i, col in enumerate(target_columns):
        rows.append({
            "target": col,
            **evaluate_predictions(y_true[:, i], y_pred[:, i], label),
        })

    rows.append({
        "target": "average",
        **evaluate_predictions(y_true.reshape(-1), y_pred.reshape(-1), label),
    })

    return pd.DataFrame(rows)


def plot_history(history, output_dir, prefix, title):
    hist = pd.DataFrame(history)

    has_val = (
        "val_mse" in hist.columns
        and not hist["val_mse"].isna().all()
    )

    for log_scale in [False, True]:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist["train_mse"], label="train mse")

        if has_val:
            plt.plot(hist["epoch"], hist["val_mse"], label="validation mse")

        if log_scale:
            plt.yscale("log")

        suffix = "_log" if log_scale else ""
        ylabel = "MSE (log scale)" if log_scale else "MSE"
        plot_title = title + " - Log Scale" if log_scale else title

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_loss{suffix}.png", dpi=160)
        plt.show()


def build_vectorized_tensors(train_df, val_df, args, device):
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
    )

    X_train = torch.tensor(
        vectorizer.fit_transform(train_df["raw_review_text"]).toarray(),
        dtype=torch.float32,
        device=device,
    )
    X_val = torch.tensor(
        vectorizer.transform(val_df["raw_review_text"]).toarray(),
        dtype=torch.float32,
        device=device,
    )

    return vectorizer, X_train, X_val


def build_group_tensors(frame, target_columns, device, target_is_1d=False):
    keys = sorted(frame["course_key"].unique())
    key_to_idx = {k: i for i, k in enumerate(keys)}

    group_index = torch.tensor(
        [key_to_idx[k] for k in frame["course_key"]],
        dtype=torch.long,
        device=device,
    )

    target_df = (
        frame[["course_key", *target_columns]]
        .drop_duplicates("course_key")
        .sort_values("course_key")
    )

    values = target_df[target_columns].values
    if target_is_1d:
        values = values.reshape(-1)

    y = torch.tensor(values, dtype=torch.float32, device=device)
    return keys, group_index, y


def unweighted_group_average(scores, group_index, num_groups):
    if scores.dim() == 1:
        score_sum = torch.zeros(num_groups, device=scores.device)
        count = torch.zeros(num_groups, device=scores.device)
        score_sum.index_add_(0, group_index, scores)
        count.index_add_(0, group_index, torch.ones(scores.shape[0], device=scores.device))
        return score_sum / (count + 1e-8)

    score_sum = torch.zeros(num_groups, scores.shape[1], device=scores.device)
    count = torch.zeros(num_groups, 1, device=scores.device)
    score_sum.index_add_(0, group_index, scores)
    count.index_add_(0, group_index, torch.ones(scores.shape[0], 1, device=scores.device))
    return score_sum / (count + 1e-8)


# ============================================================
# Model and training
# ============================================================

class SmallScoreModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, dropout=0.3):
        super().__init__()

        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.output_dim = output_dim

    def forward(self, x):
        raw = self.net(x)
        if self.output_dim == 1:
            raw = raw.squeeze(1)
        return 1.0 + 4.0 * torch.sigmoid(raw)


def train_group_average_model(
    model,
    optimizer,
    X_train,
    y_train,
    train_group_idx,
    train_group_count,
    X_val,
    args,
    val_loss_fn,
    log_name="Epoch",
):
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_scores = model(X_train)
        train_pred_avg = unweighted_group_average(
            train_scores,
            train_group_idx,
            train_group_count,
        )

        per_course_loss = ((train_pred_avg - y_train) ** 2)
        if per_course_loss.dim() == 2:
            per_course_loss = per_course_loss.mean(dim=1)

        course_counts = torch.bincount(
            train_group_idx,
            minlength=train_group_count,
        ).float().to(train_pred_avg.device)

        train_loss = (per_course_loss * course_counts).sum() / course_counts.sum()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = val_loss_fn(model, X_val)

        train_mse = float(train_loss.detach().cpu())
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val - args.min_delta:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"{log_name} {epoch:03d} | train_mse={train_mse:.4f} | val_mse={val_mse:.4f}")

        if args.early_stop and bad_epochs >= args.patience:
            print(
                f"early stopping at epoch {epoch} | "
                f"best_epoch={best_epoch} | best_val_mse={best_val:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch



def train_group_average_model_full(
    model,
    optimizer,
    X_train,
    y_train,
    train_group_idx,
    train_group_count,
    args,
    full_epochs,
    log_name="Final Epoch"
):
    """
    Final full-data training for feature generation.
    No validation, no early stopping, no best-state selection.
    This intentionally uses all available data to train the final feature extractor.
    """
    history = []

    for epoch in range(1, full_epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_scores = model(X_train)
        train_pred_avg = unweighted_group_average(
            train_scores,
            train_group_idx,
            train_group_count,
        )

        per_course_loss = ((train_pred_avg - y_train) ** 2)
        if per_course_loss.dim() == 2:
            per_course_loss = per_course_loss.mean(dim=1)

        course_counts = torch.bincount(
            train_group_idx,
            minlength=train_group_count,
        ).float().to(train_pred_avg.device)

        train_loss = (per_course_loss * course_counts).sum() / course_counts.sum()
        train_loss.backward()
        optimizer.step()

        train_mse = float(train_loss.detach().cpu())
        history.append({"epoch": epoch, "train_mse": train_mse})

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"{log_name} {epoch:03d} | train_mse={train_mse:.4f}")

    return history


def make_model_and_optimizer(input_dim, output_dim, args, device):
    model = SmallScoreModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    return model, optimizer


# ============================================================
# Main experiment
# ============================================================

def load_and_merge_main(raw_path, courses_path, target_columns):
    raw = pd.read_csv(raw_path)
    courses = pd.read_csv(courses_path)

    required_raw = ["course_name", "professor", "raw_review_text"]
    missing_raw = [c for c in required_raw if c not in raw.columns]
    if missing_raw:
        raise ValueError(f"raw file missing columns: {missing_raw}")

    missing_targets = [c for c in target_columns if c not in courses.columns]
    if missing_targets:
        raise ValueError(
            f"courses file missing target columns: {missing_targets}\n"
            f"available columns: {list(courses.columns)}"
        )

    raw = raw.copy()
    courses = courses.copy()
    raw["course_key"] = make_course_key(raw)
    courses["course_key"] = make_course_key(courses)

    course_targets = courses[["course_key", *target_columns]].copy()
    for col in target_columns:
        course_targets[col] = pd.to_numeric(course_targets[col], errors="coerce")

    course_targets = (
        course_targets
        .groupby("course_key", as_index=False)[target_columns]
        .mean()
        .dropna()
        .rename(columns={col: f"target_{col}" for col in target_columns})
    )

    target_merged_columns = [f"target_{c}" for c in target_columns]
    data = raw.merge(course_targets, on="course_key", how="inner")
    data["raw_review_text"] = data["raw_review_text"].apply(clean_text)
    data = data[data["raw_review_text"].str.len() > 0].reset_index(drop=True)

    if data.empty:
        raise ValueError("No merged reviews.")

    return data, target_merged_columns


def add_review_prediction_columns(frame, scores, target_columns, pred_suffix=""):
    out = frame.copy()

    for i, col in enumerate(target_columns):
        out[f"review_pred_{col}{pred_suffix}"] = scores[:, i]

    return out


def add_scaled_prediction_columns(df, target_columns, pred_suffix="", scaled_suffix="_scaled"):
    out = df.copy()

    for col in target_columns:
        pred_col = f"review_pred_{col}{pred_suffix}"
        target_col = f"target_{col}"
        scale_col = f"scale_{col}{pred_suffix}"
        scaled_col = f"review_pred_{col}{pred_suffix}{scaled_suffix}"

        course_stats = (
            out.groupby("course_key")
            .agg(
                pred_mean=(pred_col, "mean"),
                true_mean=(target_col, "first"),
            )
            .reset_index()
        )

        course_stats[scale_col] = (
            course_stats["true_mean"] /
            (course_stats["pred_mean"] + 1e-8)
        )

        out = out.merge(
            course_stats[["course_key", scale_col]],
            on="course_key",
            how="left",
        )

        out[scaled_col] = (out[pred_col] * out[scale_col]).clip(1.0, 5.0)

    return out


def build_oof_predictions(data, target_merged_columns, target_columns, args, device):
    fold_metrics = []
    best_epochs = []
    n_splits = min(args.n_splits, data["course_key"].nunique())

    if n_splits < 2:
        raise ValueError("Need at least 2 course groups for OOF predictions.")

    oof_scores = np.zeros((len(data), len(target_columns)), dtype=np.float32)
    groups = data["course_key"].values
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(data, groups=groups),
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"OOF Fold {fold}/{n_splits}")
        print("-" * 80)

        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)

        print(f"fold train reviews: {len(train_df)}, train courses: {train_df['course_key'].nunique()}")
        print(f"fold val reviews: {len(val_df)}, val courses: {val_df['course_key'].nunique()}")

        _, X_train, X_val = build_vectorized_tensors(train_df, val_df, args, device)

        train_group_keys, train_group_idx, y_train = build_group_tensors(
            train_df,
            target_merged_columns,
            device,
        )
        val_group_keys, val_group_idx, y_val = build_group_tensors(
            val_df,
            target_merged_columns,
            device,
        )

        model, optimizer = make_model_and_optimizer(
            input_dim=X_train.shape[1],
            output_dim=len(target_columns),
            args=args,
            device=device,
        )

        def val_loss_fn(model, X_val):
            val_scores = model(X_val)
            val_pred_avg = unweighted_group_average(
                val_scores,
                val_group_idx,
                len(val_group_keys),
            )
            return float(nn.MSELoss()(val_pred_avg, y_val).detach().cpu())

        history, best_epoch = train_group_average_model(
            model,
            optimizer,
            X_train,
            y_train,
            train_group_idx,
            len(train_group_keys),
            X_val,
            args,
            val_loss_fn,
            log_name=f"OOF Fold {fold} Epoch",
        )
        best_epochs.append(best_epoch)

        model.eval()
        with torch.no_grad():
            fold_scores = model(X_val).detach().cpu().numpy()

        oof_scores[val_idx] = fold_scores
        fold_pred_avg = unweighted_group_average(
            torch.tensor(fold_scores, device=device),
            val_group_idx,
            len(val_group_keys),
        ).cpu().numpy()

        fold_true = y_val.cpu().numpy()

        fold_result = evaluate_group_predictions(
            fold_true,
            fold_pred_avg,
            target_columns,
            f"fold_{fold}",
        )

        fold_result.insert(0, "fold", fold)

        fold_metrics.append(fold_result)

        print("\nFold Metrics")
        print(fold_result.to_string(index=False))

    fold_metrics_df = pd.concat(
        fold_metrics,
        ignore_index=True,
    )

    return oof_scores, best_epochs, fold_metrics_df


def train_final_model_on_all_data(data, target_merged_columns, target_columns, args, device, full_epochs):
    print("\n" + "=" * 80)
    print("TRAINING FINAL MAIN MODEL ON ALL DATA FOR OUTPUT FEATURES")
    print("=" * 80)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
    )

    X_all = torch.tensor(
        vectorizer.fit_transform(data["raw_review_text"]).toarray(),
        dtype=torch.float32,
        device=device,
    )

    group_keys, group_idx, y_all = build_group_tensors(
        data,
        target_merged_columns,
        device,
    )

    model, optimizer = make_model_and_optimizer(
        input_dim=X_all.shape[1],
        output_dim=len(target_columns),
        args=args,
        device=device,
    )

    history = train_group_average_model_full(
        model=model,
        optimizer=optimizer,
        X_train=X_all,
        y_train=y_all,
        train_group_idx=group_idx,
        train_group_count=len(group_keys),
        args=args,
        full_epochs=full_epochs,
        log_name="Final Epoch",
    )

    model.eval()
    with torch.no_grad():
        all_scores = model(X_all).detach().cpu().numpy()

    return history, all_scores


def build_oof_course_metrics(data, oof_scores, target_merged_columns, target_columns):
    pred_df = pd.DataFrame({"course_key": data["course_key"].values})

    for i, col in enumerate(target_columns):
        pred_df[f"pred_{col}"] = oof_scores[:, i]

    for target_col in target_merged_columns:
        pred_df[target_col] = data[target_col].values

    course_pred_df = (
        pred_df
        .groupby("course_key", as_index=False)
        .agg({
            **{f"pred_{col}": "mean" for col in target_columns},
            **{target_col: "first" for target_col in target_merged_columns},
        })
        .sort_values("course_key")
        .reset_index(drop=True)
    )

    y_true = course_pred_df[target_merged_columns].values
    y_pred = course_pred_df[[f"pred_{col}" for col in target_columns]].values

    baseline_mean = y_true.mean(axis=0)
    baseline_global = np.tile(baseline_mean, (len(y_true), 1))

    metrics = pd.concat([
        evaluate_group_predictions(
            y_true,
            baseline_global,
            target_columns,
            "main_baseline_global_mean",
        ),
        evaluate_group_predictions(
            y_true,
            y_pred,
            target_columns,
            "main_tfidf_mlp_oof",
        ),
    ], ignore_index=True)

    validation_predictions = pd.DataFrame({
        "course_key": course_pred_df["course_key"].values
    })

    for i, col in enumerate(target_columns):
        validation_predictions[f"true_{col}"] = y_true[:, i]
        validation_predictions[f"pred_{col}"] = y_pred[:, i]
        validation_predictions[f"baseline_{col}"] = baseline_global[:, i]

    return metrics, validation_predictions


def run_main_experiment(args, device):
    print("\n" + "=" * 80)
    print("MAIN EXPERIMENT: 5-fold GroupKFold OOF evaluation + final full-data output")
    print("=" * 80)

    target_columns = parse_targets(args.targets)
    data, target_merged_columns = load_and_merge_main(args.raw, args.courses, target_columns)

    print("device:", device)
    print("targets:", target_columns)
    print(f"merged reviews: {len(data)}")
    print(f"course groups: {data['course_key'].nunique()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "main_validation_metrics.csv"
    history_path = output_dir / "main_training_history.csv"
    pred_path = output_dir / "main_validation_predictions.csv"
    qualitative_path = output_dir / "main_qualitative_review_predictions.csv"
    all_qualitative_path = output_dir / "main_all_review_predictions.csv"

    # ------------------------------------------------------------
    # 1. OOF evaluation only
    #    Each review is predicted by a model that did not train on its course.
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"BUILDING {args.n_splits}-FOLD GROUP OOF REVIEW PREDICTIONS FOR MAIN EVALUATION")
    print("=" * 80)

    oof_scores, best_epochs, fold_metrics_df = build_oof_predictions(
        data=data,
        target_merged_columns=target_merged_columns,
        target_columns=target_columns,
        args=args,
        device=device,
    )

    recommended_epoch = round(np.mean(best_epochs))

    metrics, validation_predictions = build_oof_course_metrics(
        data=data,
        oof_scores=oof_scores,
        target_merged_columns=target_merged_columns,
        target_columns=target_columns,
    )

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    validation_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")

    qualitative_cols = ["course_key", "course_name", "professor", "raw_review_text"]
    if "semester" in data.columns:
        qualitative_cols.insert(3, "semester")
    if "rating" in data.columns:
        qualitative_cols.insert(4, "rating")

    qualitative_df = add_review_prediction_columns(
        data[qualitative_cols + target_merged_columns],
        oof_scores,
        target_columns,
        pred_suffix="_oof",
    )
    qualitative_df = add_scaled_prediction_columns(
        qualitative_df,
        target_columns,
        pred_suffix="_oof",
    )
    qualitative_df.to_csv(qualitative_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. Final output for Model 2
    #    Train one final model on all data, then predict all reviews.
    # ------------------------------------------------------------
    final_history, final_scores = train_final_model_on_all_data(
        data=data,
        target_merged_columns=target_merged_columns,
        target_columns=target_columns,
        args=args,
        device=device,
        full_epochs=recommended_epoch,
    )

    pd.DataFrame(final_history).to_csv(history_path, index=False, encoding="utf-8-sig")
    plot_history(
        final_history,
        output_dir,
        prefix="main_final_full_data",
        title="Final Main Model Train MSE on All Data",
    )

    all_qualitative_df = add_review_prediction_columns(
        data[qualitative_cols + target_merged_columns],
        final_scores,
        target_columns,
    )
    all_qualitative_df = add_scaled_prediction_columns(
        all_qualitative_df,
        target_columns,
    )
    all_qualitative_df.to_csv(all_qualitative_path, index=False, encoding="utf-8-sig")

    print("\n=== MAIN OOF Validation Metrics ===")
    print(metrics.to_string(index=False))
    print("\nMAIN saved:")
    print(metrics_path)
    print(history_path)
    print(pred_path)
    print(qualitative_path)
    print(all_qualitative_path)

    return metrics


# ============================================================
# Rating sanity check
# ============================================================

def load_raw_reviews_for_rating(raw_path):
    raw = pd.read_csv(raw_path)

    required = ["course_name", "professor", "rating", "raw_review_text"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"raw file missing columns for rating sanity check: {missing}")

    raw = raw.copy()
    raw["course_key"] = make_course_key(raw)
    raw["raw_review_text"] = raw["raw_review_text"].apply(clean_text)
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")

    raw = raw[
        raw["raw_review_text"].str.len().gt(0)
        & raw["rating"].notna()
    ].reset_index(drop=True)

    raw["rating"] = raw["rating"].clip(1.0, 5.0)

    if raw.empty:
        raise ValueError("No usable reviews for rating sanity check.")

    raw["course_mean_rating"] = raw.groupby("course_key")["rating"].transform("mean")

    return raw


def run_rating_sanity_check(args, device):
    print("\n" + "=" * 80)
    print("RATING SANITY CHECK: train on course mean rating, evaluate on review rating")
    print("=" * 80)

    raw = load_raw_reviews_for_rating(args.raw)

    train_df, val_df = train_test_split(
        raw,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    print(f"usable reviews: {len(raw)}")
    print(f"course groups: {raw['course_key'].nunique()}")
    print(f"rating train reviews: {len(train_df)}")
    print(f"rating validation reviews: {len(val_df)}")

    _, X_train, X_val = build_vectorized_tensors(train_df, val_df, args, device)

    y_train = torch.tensor(
        train_df["course_mean_rating"].values,
        dtype=torch.float32,
        device=device,
    )

    y_val_true = val_df["rating"].values
    baseline_course_mean = val_df["course_mean_rating"].values

    model, optimizer = make_model_and_optimizer(
        input_dim=X_train.shape[1],
        output_dim=1,
        args=args,
        device=device,
    )

    def val_loss_fn(model, X_val):
        pred = model(X_val).detach().cpu().numpy()
        return mean_squared_error(y_val_true, pred)

    history = []
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train = model(X_train)
        train_loss = criterion(pred_train, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = val_loss_fn(model, X_val)

        train_mse = float(train_loss.detach().cpu())
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val - args.min_delta:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"Rating Epoch {epoch:03d} | train_mse={train_mse:.4f} | val_mse={val_mse:.4f}")

        if args.early_stop and bad_epochs >= args.patience:
            print(f"rating early stopping at epoch {epoch} | best_val_mse={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_rating = model(X_val).detach().cpu().numpy()

    output_dir = Path(args.rating_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "rating_sanity_metrics.csv"
    history_path = output_dir / "rating_sanity_training_history.csv"
    review_pred_path = output_dir / "rating_sanity_review_predictions.csv"

    val_review_out = val_df[
        ["course_key", "course_name", "professor", "rating", "course_mean_rating", "raw_review_text"]
    ].copy()

    if "semester" in val_df.columns:
        val_review_out.insert(3, "semester", val_df["semester"].values)

    val_review_out = val_review_out.rename(columns={
        "rating": "true_rating",
        "course_mean_rating": "baseline_course_mean_rating",
    })

    val_review_out["pred_rating"] = pred_rating

    course_stats = (
        val_review_out.groupby("course_key")
        .agg(
            pred_mean=("pred_rating", "mean"),
            true_mean=("baseline_course_mean_rating", "first"),
        )
        .reset_index()
    )

    course_stats["scale"] = (
        course_stats["true_mean"] /
        (course_stats["pred_mean"] + 1e-8)
    )

    val_review_out = val_review_out.merge(
        course_stats[["course_key", "scale"]],
        on="course_key",
        how="left",
    )

    val_review_out["pred_rating_scaled"] = (
        val_review_out["pred_rating"] *
        val_review_out["scale"]
    ).clip(1.0, 5.0)

    metrics = pd.DataFrame([
        evaluate_predictions(
            y_val_true,
            baseline_course_mean,
            "rating_baseline_course_mean",
        ),
        evaluate_predictions(
            y_val_true,
            pred_rating,
            "rating_review_text_model",
        ),
        evaluate_predictions(
            y_val_true,
            val_review_out["pred_rating_scaled"].values,
            "rating_review_text_model_scaled_to_course_mean",
        ),
    ])

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    val_review_out.to_csv(review_pred_path, index=False, encoding="utf-8-sig")

    plot_history(
        history,
        output_dir,
        prefix="rating_sanity",
        title="Rating Sanity Check: Course Mean vs Review Text Model",
    )

    print("\n=== RATING Sanity Check Metrics ===")
    print(metrics.to_string(index=False))
    print("\nRATING saved:")
    print(metrics_path)
    print(history_path)
    print(review_pred_path)

    return metrics


# ============================================================
# CLI
# ============================================================

def add_common_training_args(parser):
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--final-epochs", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--early-stop", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-4)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        default="scripts/examples/train_course_attribute_model/input/raw_everytime_reviews.csv",
    )
    
    parser.add_argument(
        "--courses",
        default="scripts/examples/train_course_attribute_model/input/courses.csv",
    )
    
    parser.add_argument(
        "--output-dir",
        default="scripts/examples/train_course_attribute_model/output/main_model",
    )
    
    parser.add_argument(
        "--rating-output-dir",
        default="scripts/examples/train_course_attribute_model/output/rating_sanity",
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--run-rating-check", action="store_true", default=True)
    parser.add_argument("--no-rating-check", dest="run_rating_check", action="store_false")

    add_common_training_args(parser)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_main_experiment(args, device)

    if args.run_rating_check:
        run_rating_sanity_check(args, device)
    else:
        print("\nRating sanity check skipped. Main outputs are unaffected.")


if __name__ == "__main__":
    main()
