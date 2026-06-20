import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGET_COLUMNS = [
    "rating",
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]


# ============================================================
# Basic utilities
# ============================================================

def semester_to_number(value):
    text = str(value).strip()
    if not text:
        return np.nan

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

    return np.nan


def number_to_semester_label(num):
    year = int(num // 2)
    term = int(num % 2) + 1
    return f"{year}-{term}"


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


def evaluate_predictions(y_true, y_pred, target_columns, model_name, k_history=None):
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for i, target in enumerate(target_columns):
        rows.append({
            "k_history": k_history,
            "target": target,
            "model": model_name,
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "RMSE": math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "Pearson_r": safe_pearson(y_true[:, i], y_pred[:, i]),
        })

    rows.append({
        "k_history": k_history,
        "target": "average",
        "model": model_name,
        "MAE": mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)),
        "RMSE": math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "Pearson_r": safe_pearson(y_true.reshape(-1), y_pred.reshape(-1)),
    })
    return pd.DataFrame(rows)


# ============================================================
# Input loading
# ============================================================

def infer_prediction_columns(frame, target_columns, use_scaled_predictions=True):
    pred_columns = []

    for target in target_columns:
        if target == "rating":
            candidate = "rating"
        else:
            raw_candidate = f"review_pred_{target}"
            scaled_candidate = f"review_pred_{target}_scaled"

            if use_scaled_predictions and scaled_candidate in frame.columns:
                candidate = scaled_candidate
            else:
                candidate = raw_candidate

        if candidate not in frame.columns:
            raise ValueError(
                f"Missing column: {candidate}\n"
                f"Available columns: {list(frame.columns)}"
            )

        pred_columns.append(candidate)

    return pred_columns


def load_review_predictions(path, target_columns, use_scaled_predictions=True):
    frame = pd.read_csv(path)

    required = ["course_key", "semester"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"review prediction file missing columns: {missing}")

    pred_columns = infer_prediction_columns(
        frame,
        target_columns,
        use_scaled_predictions=use_scaled_predictions,
    )

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
        raise ValueError("No usable review predictions after filtering semester/prediction columns.")

    print(f"filter summary: raw={before}, valid_semester={after_semester}, valid_predictions={after_pred}")
    print("prediction columns used:")
    for target, col in zip(target_columns, pred_columns):
        print(f"  {target}: {col}")

    return frame, pred_columns


def build_course_semester_table(review_df, target_columns, pred_columns):
    """
    Convert review-level BERT outputs into course-semester-level inputs.
    Each course-semester value is the mean of review-level BERT predictions in that semester.
    """
    agg_dict = {pred_col: "mean" for pred_col in pred_columns}

    if "raw_review_text" in review_df.columns:
        agg_dict["raw_review_text"] = "count"
    if "course_name" in review_df.columns:
        agg_dict["course_name"] = "first"
    if "professor" in review_df.columns:
        agg_dict["professor"] = "first"

    table = (
        review_df
        .groupby(["course_key", "semester_num"], as_index=False)
        .agg(agg_dict)
    )

    if "raw_review_text" in table.columns:
        table = table.rename(columns={"raw_review_text": "review_count"})
    else:
        count_df = (
            review_df.groupby(["course_key", "semester_num"], as_index=False)
            .size()
            .rename(columns={"size": "review_count"})
        )
        table = table.merge(count_df, on=["course_key", "semester_num"], how="left")

    rename_map = {
        pred_col: f"semester_avg_{target}"
        for target, pred_col in zip(target_columns, pred_columns)
    }
    table = table.rename(columns=rename_map)
    table["semester_label"] = table["semester_num"].apply(number_to_semester_label)

    return table


# ============================================================
# Flatten + summary samples
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

                if len(hist) >= 2:
                    row[f"trend_last_diff_{target}"] = prev_value - float(hist.iloc[-2][col])
                else:
                    row[f"trend_last_diff_{target}"] = 0.0

                # Single-output baselines. These are saved once, not repeated for every model.
                row[f"pred_last_value_{target}"] = prev_value
                row[f"pred_historical_mean_{target}"] = hist_mean
                row[f"true_{target}"] = float(cur[col])

            rows.append(row)

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("No model2 samples. Need at least two valid semesters per course.")

    return samples


def get_flatten_feature_columns(target_columns, k_history, use_last_diff_trend=True):
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

    target_y_columns = [f"true_{target}" for target in target_columns]
    return feature_columns, target_y_columns


def split_samples(samples, val_size, seed, group_split=True):
    if group_split:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(samples, groups=samples["course_key"]))
        return (
            samples.iloc[train_idx].reset_index(drop=True),
            samples.iloc[val_idx].reset_index(drop=True),
        )

    train_df, val_df = train_test_split(
        samples,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_tabular_model(model_name, seed, ridge_alpha, mlp_max_iter):
    if model_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=ridge_alpha)),
        ])

    if model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                random_state=seed,
            )),
        ])

    raise ValueError(f"Unknown tabular model: {model_name}")


def run_tabular_model(samples, target_columns, k_history, model_name, args, feature_columns, target_y_columns):
    train_df, val_df = split_samples(samples, args.val_size, args.seed, args.group_split)

    X_train = train_df[feature_columns].values
    y_train = train_df[target_y_columns].values
    X_val = val_df[feature_columns].values
    y_val = val_df[target_y_columns].values

    model = make_tabular_model(model_name, args.seed, args.ridge_alpha, args.mlp_max_iter)
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_val), 1.0, 5.0)

    metrics = evaluate_predictions(
        y_val,
        pred,
        target_columns,
        f"model2_{model_name}_k{k_history}",
        k_history,
    )

    pred_df = val_df.copy()
    for i, target in enumerate(target_columns):
        pred_df[f"model2_pred_{target}"] = pred[:, i]

    return metrics, pred_df


# ============================================================
# Baseline metrics, saved once
# ============================================================

def evaluate_semester_bert_mean_baselines(samples, target_columns, output_dir, args):
    target_y_columns = [f"true_{target}" for target in target_columns]

    _, val_df = split_samples(samples, args.val_size, args.seed, args.group_split)

    y_val = val_df[target_y_columns].values
    pred_last = val_df[[f"pred_last_value_{target}" for target in target_columns]].values
    pred_hist = val_df[[f"pred_historical_mean_{target}" for target in target_columns]].values

    metrics = pd.concat([
        evaluate_predictions(
            y_val,
            pred_last,
            target_columns,
            "semester_bert_last_mean",
            "baseline",
        ),
        evaluate_predictions(
            y_val,
            pred_hist,
            target_columns,
            "semester_bert_historical_mean",
            "baseline",
        ),
    ], ignore_index=True)

    pred_df = val_df.copy()
    for i, target in enumerate(target_columns):
        pred_df[f"semester_bert_last_mean_{target}"] = pred_last[:, i]
        pred_df[f"semester_bert_historical_mean_{target}"] = pred_hist[:, i]

    metrics.to_csv(output_dir / "model2_semester_bert_mean_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "model2_semester_bert_mean_baseline_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics


# ============================================================
# GRU sequence model
# ============================================================

def build_sequence_samples(course_semester_df, target_columns, min_gap=1, max_gap=None):
    rows = []
    sequences = []
    targets = []
    value_cols = [f"semester_avg_{target}" for target in target_columns]

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

            seq = []
            prev_semester = None

            for _, h in hist.iterrows():
                if prev_semester is None:
                    local_gap = 0.0
                else:
                    local_gap = float(h["semester_num"] - prev_semester)
                prev_semester = h["semester_num"]

                seq_row = [float(h[col]) for col in value_cols]
                seq_row.extend([
                    float(h["review_count"]),
                    local_gap,
                ])
                seq.append(seq_row)

            y = [float(cur[col]) for col in value_cols]

            meta = {
                "course_key": course_key,
                "prev_semester": prev["semester_label"],
                "target_semester": cur["semester_label"],
                "prev_semester_num": prev["semester_num"],
                "target_semester_num": cur["semester_num"],
                "semester_gap": gap,
                "history_length": len(hist),
            }

            if "course_name" in group.columns:
                meta["course_name"] = cur.get("course_name", "")
            if "professor" in group.columns:
                meta["professor"] = cur.get("professor", "")

            sequences.append(np.asarray(seq, dtype=np.float32))
            targets.append(np.asarray(y, dtype=np.float32))
            rows.append(meta)

    if not sequences:
        raise ValueError("No sequence samples. Need at least two valid semesters per course.")

    return pd.DataFrame(rows), sequences, np.vstack(targets)


def split_sequence_samples(meta_df, val_size, seed, group_split=True):
    indices = np.arange(len(meta_df))

    if group_split:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(meta_df, groups=meta_df["course_key"]))
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )

    return train_idx, val_idx


def run_gru_model(meta_df, sequences, y, target_columns, args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        raise ImportError("GRU model requires torch. Install: pip install torch") from e

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_idx, val_idx = split_sequence_samples(
        meta_df,
        args.val_size,
        args.seed,
        args.group_split,
    )

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    all_train_steps = np.vstack(train_sequences)
    x_mean = all_train_steps.mean(axis=0, keepdims=True)
    x_std = all_train_steps.std(axis=0, keepdims=True) + 1e-8

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    train_sequences = [(s - x_mean) / x_std for s in train_sequences]
    val_sequences = [(s - x_mean) / x_std for s in val_sequences]
    y_train_scaled = (y_train - y_mean) / y_std

    class SeqDataset(Dataset):
        def __init__(self, seqs, labels=None):
            self.seqs = seqs
            self.labels = labels

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            if self.labels is None:
                return self.seqs[idx]
            return self.seqs[idx], self.labels[idx]

    def collate_train(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        max_len = int(lengths.max())
        feat_dim = seqs[0].shape[1]

        x = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.float32)

        yb = torch.tensor(np.vstack(labels), dtype=torch.float32)
        return x, lengths, yb

    def collate_pred(batch):
        lengths = torch.tensor([len(s) for s in batch], dtype=torch.long)
        max_len = int(lengths.max())
        feat_dim = batch[0].shape[1]

        x = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
        for i, s in enumerate(batch):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.float32)

        return x, lengths

    class GRURegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h = self.gru(packed)
            h = h[-1]
            return self.head(self.dropout(h))

    train_loader = DataLoader(
        SeqDataset(train_sequences, y_train_scaled),
        batch_size=args.gru_batch_size,
        shuffle=True,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        SeqDataset(val_sequences),
        batch_size=args.gru_batch_size,
        shuffle=False,
        collate_fn=collate_pred,
    )

    input_dim = train_sequences[0].shape[1]
    model = GRURegressor(
        input_dim,
        args.gru_hidden_dim,
        len(target_columns),
        args.gru_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.gru_lr,
        weight_decay=args.gru_weight_decay,
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.gru_epochs + 1):
        model.train()
        train_losses = []

        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb, lengths)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gru_grad_clip)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        preds_scaled = []

        with torch.no_grad():
            for xb, lengths in val_loader:
                xb = xb.to(device)
                lengths = lengths.to(device)
                pred = model(xb, lengths).detach().cpu().numpy()
                preds_scaled.append(pred)

        pred_val = np.vstack(preds_scaled) * y_std + y_mean
        pred_val = np.clip(pred_val, 1.0, 5.0)

        val_mse = mean_squared_error(y_val, pred_val)
        train_mse = float(np.mean(train_losses))
        history.append({
            "epoch": epoch,
            "train_scaled_mse": train_mse,
            "val_mse": val_mse,
        })

        if epoch == 1 or epoch % args.gru_log_every == 0:
            print(f"GRU epoch {epoch:03d} | train_scaled_mse={train_mse:.4f} | val_mse={val_mse:.4f}")

        if val_mse < best_val - args.gru_min_delta:
            best_val = val_mse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.gru_patience:
            print(f"GRU early stopping at epoch {epoch} | best_val_mse={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, lengths in val_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            pred = model(xb, lengths).detach().cpu().numpy()
            preds_scaled.append(pred)

    pred_val = np.vstack(preds_scaled) * y_std + y_mean
    pred_val = np.clip(pred_val, 1.0, 5.0)

    val_meta = meta_df.iloc[val_idx].reset_index(drop=True).copy()
    for i, target in enumerate(target_columns):
        val_meta[f"true_{target}"] = y_val[:, i]
        val_meta[f"model2_pred_{target}"] = pred_val[:, i]

    return pred_val, y_val, val_meta, pd.DataFrame(history)


# ============================================================
# Experiment runner
# ============================================================

def parse_model_configs(value):
    """
    Format examples:
      ridge:1,ridge:3,mlp:1,gru:all
    """
    configs = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                f"Invalid model config: {item}. "
                f"Use format like ridge:1,mlp:3,gru:all"
            )

        model_name, history_value = item.split(":", 1)
        model_name = model_name.strip()
        history_value = history_value.strip()

        if model_name in ["ridge", "mlp"]:
            configs.append((model_name, int(history_value)))
        elif model_name == "gru":
            if history_value != "all":
                raise ValueError("GRU config must be gru:all")
            configs.append((model_name, "all"))
        else:
            raise ValueError(f"Unknown model config: {item}")

    return configs


def evaluate_all_models(course_semester, target_columns, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    model_configs = parse_model_configs(args.model_configs)

    # Baseline from semester-level BERT averages is saved once only.
    baseline_samples = build_flatten_summary_samples(
        course_semester,
        target_columns,
        k_history=1,
        min_gap=args.min_gap,
        max_gap=None if args.max_gap == 0 else args.max_gap,
    )
    baseline_metrics = evaluate_semester_bert_mean_baselines(
        baseline_samples,
        target_columns,
        output_dir,
        args,
    )

    if args.include_baselines_in_summary:
        all_metrics.append(baseline_metrics)

    for model_name, history_value in model_configs:
        if model_name in ["ridge", "mlp"]:
            k_history = int(history_value)
            print("\n" + "=" * 80)
            print(f"Evaluating {model_name} with k_history={k_history}")
            print("=" * 80)

            samples = build_flatten_summary_samples(
                course_semester,
                target_columns,
                k_history=k_history,
                min_gap=args.min_gap,
                max_gap=None if args.max_gap == 0 else args.max_gap,
            )
            feature_columns, target_y_columns = get_flatten_feature_columns(
                target_columns,
                k_history=k_history,
                use_last_diff_trend=args.use_last_diff_trend,
            )

            print(f"samples: {len(samples)}, course groups: {samples['course_key'].nunique()}")
            print(f"features: {len(feature_columns)}")

            samples.to_csv(
                output_dir / f"model2_samples_{model_name}_k{k_history}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            with open(output_dir / f"model2_feature_columns_{model_name}_k{k_history}.txt", "w", encoding="utf-8") as f:
                for col in feature_columns:
                    f.write(col + "\n")

            metrics, pred_df = run_tabular_model(
                samples,
                target_columns,
                k_history,
                model_name,
                args,
                feature_columns,
                target_y_columns,
            )

            all_metrics.append(metrics)
            pred_df.to_csv(
                output_dir / f"model2_{model_name}_k{k_history}_validation_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(metrics.to_string(index=False))

        elif model_name == "gru":
            print("\n" + "=" * 80)
            print("Evaluating GRU sequence model")
            print("=" * 80)

            meta_df, sequences, y = build_sequence_samples(
                course_semester,
                target_columns,
                min_gap=args.min_gap,
                max_gap=None if args.max_gap == 0 else args.max_gap,
            )

            pred, y_val, pred_df, history = run_gru_model(
                meta_df,
                sequences,
                y,
                target_columns,
                args,
            )

            metrics = evaluate_predictions(
                y_val,
                pred,
                target_columns,
                "model2_gru_all",
                "all",
            )

            all_metrics.append(metrics)
            pred_df.to_csv(
                output_dir / "model2_gru_validation_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
            history.to_csv(
                output_dir / "model2_gru_training_history.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(metrics.to_string(index=False))

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "model2_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    return metrics, baseline_metrics


# ============================================================
# CLI
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-predictions",
        default="scripts/examples/train_course_attribute_model/output/bert_course_attribute_model/bert_all_review_predictions.csv",
        help="Model1 BERT all-review prediction CSV.",
    )
    parser.add_argument(
        "--targets",
        default="rating,workload_label,teamwork_load_label,grading_strictness_label",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_model2_scaled_bert_comparison",
    )

    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--group-split", action="store_true", default=True)
    parser.add_argument("--random-split", dest="group_split", action="store_false")
    parser.add_argument("--no-last-diff-trend", dest="use_last_diff_trend", action="store_false")
    parser.set_defaults(use_last_diff_trend=True)

    parser.add_argument(
        "--use-scaled-predictions",
        action="store_true",
        default=True,
        help="Use review_pred_*_scaled columns for non-rating targets when available.",
    )
    parser.add_argument(
        "--use-raw-predictions",
        dest="use_scaled_predictions",
        action="store_false",
        help="Use unscaled review_pred_* columns.",
    )

    parser.add_argument(
        "--model-configs",
        default="ridge:1,ridge:3,ridge:5,mlp:1,mlp:3,gru:all",
        help="Exactly the model configs to evaluate. Example: ridge:1,ridge:3,mlp:1,gru:all",
    )
    parser.add_argument(
        "--include-baselines-in-summary",
        action="store_true",
        default=False,
        help="If set, include semester_bert_last_mean and semester_bert_historical_mean in model2_comparison_metrics.csv. "
             "They are always saved separately once.",
    )

    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-max-iter", type=int, default=2000)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gru-hidden-dim", type=int, default=32)
    parser.add_argument("--gru-dropout", type=float, default=0.2)
    parser.add_argument("--gru-lr", type=float, default=1e-3)
    parser.add_argument("--gru-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gru-epochs", type=int, default=300)
    parser.add_argument("--gru-batch-size", type=int, default=32)
    parser.add_argument("--gru-patience", type=int, default=30)
    parser.add_argument("--gru-min-delta", type=float, default=1e-4)
    parser.add_argument("--gru-grad-clip", type=float, default=1.0)
    parser.add_argument("--gru-log-every", type=int, default=25)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()

    target_columns = parse_targets(args.targets)

    review_df, pred_columns = load_review_predictions(
        args.review_predictions,
        target_columns,
        use_scaled_predictions=args.use_scaled_predictions,
    )

    print("loaded review predictions:", len(review_df))
    print("course groups:", review_df["course_key"].nunique())
    print("targets:", target_columns)

    course_semester = build_course_semester_table(
        review_df,
        target_columns,
        pred_columns,
    )

    print("course-semester rows:", len(course_semester))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    course_semester.to_csv(
        output_dir / "model2_course_semester_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics, baseline_metrics = evaluate_all_models(
        course_semester,
        target_columns,
        args,
    )

    print("\n=== Semester-level BERT mean baselines, saved once ===")
    print(baseline_metrics.to_string(index=False))

    print("\n=== Model2 comparison metrics ===")
    print(metrics.to_string(index=False))

    print("\nsaved:")
    print(output_dir / "model2_semester_bert_mean_baseline_metrics.csv")
    print(output_dir / "model2_comparison_metrics.csv")


if __name__ == "__main__":
    main()
