#실제로 bert fine-tuning으로 5fold 후 예측값 생성

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_TARGET_COLUMNS = [
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]


# ============================================================
# Utilities
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


# ============================================================
# Data
# ============================================================

def load_and_merge_data(raw_path, courses_path, target_columns):
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

    target_columns_merged = [f"target_{c}" for c in target_columns]

    data = raw.merge(course_targets, on="course_key", how="inner")
    data["raw_review_text"] = data["raw_review_text"].apply(clean_text)
    data = data[data["raw_review_text"].str.len() > 0].reset_index(drop=True)

    if data.empty:
        raise ValueError("No merged reviews. Check course_name/professor matching.")

    return data, target_columns_merged


def build_course_target_table(frame, target_columns_merged):
    target_df = (
        frame[["course_key", *target_columns_merged]]
        .drop_duplicates("course_key")
        .sort_values("course_key")
        .reset_index(drop=True)
    )
    return target_df


# ============================================================
# BERT model
# ============================================================

class ReviewDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.texts[idx]
        return self.texts[idx], self.labels[idx]


class BertRegressor(nn.Module):
    def __init__(self, model_name, output_dim, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.bert(**kwargs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]

        raw = self.head(self.dropout(pooled))
        return 1.0 + 4.0 * torch.sigmoid(raw)


def make_collate_fn(tokenizer, max_length, has_labels):
    def collate(batch):
        if has_labels:
            texts, labels = zip(*batch)
        else:
            texts = batch
            labels = None

        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if labels is None:
            return enc

        labels = torch.tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32)
        return enc, labels

    return collate


def repeated_course_targets(frame, target_columns_merged):
    return frame[target_columns_merged].values.astype(np.float32)


def train_bert_model(train_df, val_df, target_columns_merged, args, device, fold_name="Fold"):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    train_labels = repeated_course_targets(train_df, target_columns_merged)
    val_labels = repeated_course_targets(val_df, target_columns_merged)

    train_ds = ReviewDataset(train_df["raw_review_text"].values, train_labels)
    val_ds = ReviewDataset(val_df["raw_review_text"].values, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bert_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.bert_max_length, has_labels=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bert_batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.bert_max_length, has_labels=True),
    )

    model = BertRegressor(
        model_name=args.bert_model,
        output_dim=len(target_columns_merged),
        dropout=args.bert_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.bert_lr,
        weight_decay=args.bert_weight_decay,
    )

    total_steps = max(len(train_loader) * args.bert_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.bert_warmup_ratio),
        num_training_steps=total_steps,
    )

    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.bert_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for step, (enc, labels) in enumerate(train_loader, start=1):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(**enc)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.bert_grad_clip)
            optimizer.step()
            scheduler.step()

            batch_size = labels.shape[0]
            train_total += float(loss.detach().cpu()) * batch_size
            train_count += batch_size

            if step == 1 or step % args.bert_log_every == 0 or step == len(train_loader):
                print(
                    f"{fold_name} epoch {epoch}/{args.bert_epochs} "
                    f"| step {step}/{len(train_loader)} "
                    f"| train_mse={train_total / max(train_count, 1):.4f}"
                )

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for enc, labels in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                pred = model(**enc)
                loss = criterion(pred, labels)
                batch_size = labels.shape[0]
                val_total += float(loss.detach().cpu()) * batch_size
                val_count += batch_size

        train_mse = train_total / max(train_count, 1)
        val_mse = val_total / max(val_count, 1)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        print(
            f"{fold_name} epoch {epoch} done "
            f"| train_mse={train_mse:.4f} | val_mse={val_mse:.4f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, tokenizer, history, best_epoch


def train_bert_full_model(data, target_columns_merged, args, device, full_epochs):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    labels = repeated_course_targets(data, target_columns_merged)
    train_ds = ReviewDataset(data["raw_review_text"].values, labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bert_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.bert_max_length, has_labels=True),
    )

    model = BertRegressor(
        model_name=args.bert_model,
        output_dim=len(target_columns_merged),
        dropout=args.bert_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.bert_lr,
        weight_decay=args.bert_weight_decay,
    )

    total_steps = max(len(train_loader) * full_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.bert_warmup_ratio),
        num_training_steps=total_steps,
    )

    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, full_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for step, (enc, labels) in enumerate(train_loader, start=1):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(**enc)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.bert_grad_clip)
            optimizer.step()
            scheduler.step()

            batch_size = labels.shape[0]
            train_total += float(loss.detach().cpu()) * batch_size
            train_count += batch_size

            if step == 1 or step % args.bert_log_every == 0 or step == len(train_loader):
                print(
                    f"Final epoch {epoch}/{full_epochs} "
                    f"| step {step}/{len(train_loader)} "
                    f"| train_mse={train_total / max(train_count, 1):.4f}"
                )

        train_mse = train_total / max(train_count, 1)
        history.append({"epoch": epoch, "train_mse": train_mse})
        print(f"Final epoch {epoch} done | train_mse={train_mse:.4f}")

    return model, tokenizer, history


def predict_bert(model, tokenizer, frame, args, device):
    ds = ReviewDataset(frame["raw_review_text"].values, labels=None)
    loader = DataLoader(
        ds,
        batch_size=args.bert_batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.bert_max_length, has_labels=False),
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for enc in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = model(**enc)
            preds.append(pred.detach().cpu().numpy())

    return np.clip(np.vstack(preds), 1.0, 5.0)


# ============================================================
# Output helpers
# ============================================================

def add_prediction_columns(frame, scores, target_columns, suffix=""):
    out = frame.copy()
    for i, col in enumerate(target_columns):
        out[f"review_pred_{col}{suffix}"] = scores[:, i]
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

        course_stats[scale_col] = course_stats["true_mean"] / (course_stats["pred_mean"] + 1e-8)

        out = out.merge(
            course_stats[["course_key", scale_col]],
            on="course_key",
            how="left",
        )

        out[scaled_col] = (out[pred_col] * out[scale_col]).clip(1.0, 5.0)

    return out


def build_course_metrics(data, scores, target_columns, target_columns_merged, model_name):
    pred_df = pd.DataFrame({"course_key": data["course_key"].values})

    for i, col in enumerate(target_columns):
        pred_df[f"pred_{col}"] = scores[:, i]

    for col in target_columns_merged:
        pred_df[col] = data[col].values

    course_pred = (
        pred_df.groupby("course_key", as_index=False)
        .agg({
            **{f"pred_{col}": "mean" for col in target_columns},
            **{col: "first" for col in target_columns_merged},
        })
        .sort_values("course_key")
        .reset_index(drop=True)
    )

    y_true = course_pred[target_columns_merged].values
    y_pred = course_pred[[f"pred_{col}" for col in target_columns]].values

    metrics = evaluate_group_predictions(y_true, y_pred, target_columns, model_name)

    predictions = pd.DataFrame({"course_key": course_pred["course_key"].values})
    for i, col in enumerate(target_columns):
        predictions[f"true_{col}"] = y_true[:, i]
        predictions[f"pred_{col}"] = y_pred[:, i]

    return metrics, predictions


# ============================================================
# Main
# ============================================================

def run_experiment(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    target_columns = parse_targets(args.targets)

    data, target_columns_merged = load_and_merge_data(args.raw, args.courses, target_columns)

    print("device:", device)
    print("model:", args.bert_model)
    print("targets:", target_columns)
    print("merged reviews:", len(data))
    print("course groups:", data["course_key"].nunique())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_path = output_dir / "bert_fold_metrics.csv"
    fold_predictions_path = output_dir / "bert_fold_course_predictions.csv"
    qualitative_path = output_dir / "bert_qualitative_review_predictions.csv"
    all_predictions_path = output_dir / "bert_all_review_predictions.csv"
    final_history_path = output_dir / "bert_final_training_history.csv"

    # ------------------------------------------------------------
    # 1. GroupKFold quality evaluation
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"BUILDING {args.n_splits}-FOLD GROUP BERT PREDICTIONS FOR QUALITY CHECK")
    print("=" * 80)

    n_splits = min(args.n_splits, data["course_key"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    groups = data["course_key"].values

    oof_scores = np.zeros((len(data), len(target_columns)), dtype=np.float32)
    fold_metric_frames = []
    fold_prediction_frames = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(data, groups=groups), start=1):
        print("\n" + "-" * 80)
        print(f"BERT Fold {fold}/{n_splits}")
        print("-" * 80)

        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)

        print(f"train reviews={len(train_df)}, train courses={train_df['course_key'].nunique()}")
        print(f"val reviews={len(val_df)}, val courses={val_df['course_key'].nunique()}")

        model, tokenizer, history, best_epoch = train_bert_model(
            train_df=train_df,
            val_df=val_df,
            target_columns_merged=target_columns_merged,
            args=args,
            device=device,
            fold_name=f"BERT Fold {fold}",
        )
        best_epochs.append(best_epoch)

        fold_scores = predict_bert(model, tokenizer, val_df, args, device)
        oof_scores[val_idx] = fold_scores

        fold_metrics, fold_predictions = build_course_metrics(
            data=val_df,
            scores=fold_scores,
            target_columns=target_columns,
            target_columns_merged=target_columns_merged,
            model_name=f"bert_fold_{fold}",
        )
        fold_metrics.insert(0, "fold", fold)
        fold_predictions.insert(0, "fold", fold)

        fold_metric_frames.append(fold_metrics)
        fold_prediction_frames.append(fold_predictions)

        print("\nFold metrics")
        print(fold_metrics.to_string(index=False))

    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    fold_predictions_df = pd.concat(fold_prediction_frames, ignore_index=True)

    fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    fold_predictions_df.to_csv(fold_predictions_path, index=False, encoding="utf-8-sig")

    qualitative_cols = ["course_key", "course_name", "professor", "raw_review_text"]
    if "semester" in data.columns:
        qualitative_cols.insert(3, "semester")
    if "rating" in data.columns:
        qualitative_cols.insert(4, "rating")

    qualitative_df = add_prediction_columns(
        data[qualitative_cols + target_columns_merged],
        oof_scores,
        target_columns,
        suffix="_oof",
    )
    qualitative_df = add_scaled_prediction_columns(
        qualitative_df,
        target_columns,
        pred_suffix="_oof",
    )
    qualitative_df.to_csv(qualitative_path, index=False, encoding="utf-8-sig")

    recommended_epoch = max(1, round(float(np.mean(best_epochs))))
    print("\nOOF best epochs:", best_epochs)
    print("Recommended full-data epochs:", recommended_epoch)

    # ------------------------------------------------------------
    # 2. Final full-data model for Model2 input
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING FINAL BERT MODEL ON ALL DATA FOR MODEL2 INPUT")
    print("=" * 80)

    full_epochs = args.final_epochs if args.final_epochs > 0 else recommended_epoch

    final_model, final_tokenizer, final_history = train_bert_full_model(
        data=data,
        target_columns_merged=target_columns_merged,
        args=args,
        device=device,
        full_epochs=full_epochs,
    )

    pd.DataFrame(final_history).to_csv(final_history_path, index=False, encoding="utf-8-sig")

    final_scores = predict_bert(final_model, final_tokenizer, data, args, device)

    all_predictions_df = add_prediction_columns(
        data[qualitative_cols + target_columns_merged],
        final_scores,
        target_columns,
    )
    all_predictions_df = add_scaled_prediction_columns(
        all_predictions_df,
        target_columns,
    )
    all_predictions_df.to_csv(all_predictions_path, index=False, encoding="utf-8-sig")

    print("\n=== BERT Fold Metrics ===")
    print(fold_metrics_df.to_string(index=False))

    print("\nsaved:")
    print(fold_metrics_path)
    print(fold_predictions_path)
    print(qualitative_path)
    print(all_predictions_path)
    print(final_history_path)


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
        default="scripts/examples/train_course_attribute_model/output/bert_course_attribute_model",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGET_COLUMNS),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--bert-model", default="klue/bert-base")
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=0,
        help="0 means use the average best epoch from GroupKFold.",
    )
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--bert-lr", type=float, default=2e-5)
    parser.add_argument("--bert-weight-decay", type=float, default=0.01)
    parser.add_argument("--bert-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--bert-dropout", type=float, default=0.1)
    parser.add_argument("--bert-grad-clip", type=float, default=1.0)
    parser.add_argument("--bert-log-every", type=int, default=50)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    run_experiment(get_args())
