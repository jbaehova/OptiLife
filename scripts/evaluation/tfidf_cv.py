"""
MIL CV Experiment — upstream SmallScoreModel 하이퍼파라미터 서치.

두 축 GroupKFold(leave-course-out, leave-professor-out) ×
baseline(고정 HP) + N-iter 랜덤 서치.

사용법:
    python mil_cv_experiment.py \
        --raw ../../references/OptiLife/data/csv/raw_everytime_reviews.csv \
        --courses ../../references/OptiLife/data/csv/courses.csv

    # 빠른 스모크 테스트
    python mil_cv_experiment.py --n-iter 2 --n-splits 2 ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy.stats import pearsonr, ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterSampler


# ════════════════════════════════════════════════════════════════
# 1. Configuration
# ════════════════════════════════════════════════════════════════

TARGETS = [
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]

UPSTREAM_DEFAULTS: dict = dict(
    hidden_dim=64,
    dropout=0.3,
    lr=5e-4,
    weight_decay=5e-4,
    max_features=5000,
    min_df=2,
    ngram_range=(1, 2),
)

SEARCH_SPACE: dict = dict(
    hidden_dim=[32, 64, 128],
    max_features=[1000, 3000, 5000],
    dropout=[0.1, 0.3, 0.5],
    lr=[1e-3, 5e-4, 3e-4],
    min_df=[1, 2, 3],
    ngram_range=[(1, 1), (1, 2)],
)

FIXED_TRAINING: dict = dict(
    epochs=800,
    patience=80,
    min_delta=1e-4,
    weight_decay=5e-4,
)

SplitAxis = Literal["course", "professor"]


# ════════════════════════════════════════════════════════════════
# 2. Model — upstream SmallScoreModel 그대로
# ════════════════════════════════════════════════════════════════

class SmallScoreModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dim: int = 64, dropout: float = 0.3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.output_dim == 1:
            raw = raw.squeeze(1)
        return 1.0 + 4.0 * torch.sigmoid(raw)


def group_average(
    scores: torch.Tensor,
    group_idx: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    if scores.dim() == 1:
        s = torch.zeros(n_groups, device=scores.device)
        c = torch.zeros(n_groups, device=scores.device)
        s.index_add_(0, group_idx, scores)
        c.index_add_(0, group_idx, torch.ones_like(scores))
        return s / (c + 1e-8)

    s = torch.zeros(n_groups, scores.shape[1], device=scores.device)
    c = torch.zeros(n_groups, 1, device=scores.device)
    s.index_add_(0, group_idx, scores)
    c.index_add_(0, group_idx, torch.ones(len(scores), 1, device=scores.device))
    return s / (c + 1e-8)


# ════════════════════════════════════════════════════════════════
# 3. Data
# ════════════════════════════════════════════════════════════════

def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _course_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["course_name"].astype(str).str.strip()
        + "__"
        + df["professor"].fillna("").astype(str).str.strip()
    )


def load_data(
    raw_path: str, courses_path: str, targets: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(raw_path)
    courses = pd.read_csv(courses_path)

    raw = raw.copy()
    courses = courses.copy()
    raw["course_key"] = _course_key(raw)
    courses["course_key"] = _course_key(courses)

    ct = courses[["course_key", *targets]].copy()
    for col in targets:
        ct[col] = pd.to_numeric(ct[col], errors="coerce")

    target_cols = [f"target_{c}" for c in targets]
    ct = (
        ct.groupby("course_key", as_index=False)[targets]
        .mean()
        .dropna()
        .rename(columns={c: f"target_{c}" for c in targets})
    )

    data = raw.merge(ct, on="course_key", how="inner")
    data["raw_review_text"] = data["raw_review_text"].apply(_clean)
    data = data[data["raw_review_text"].str.len() > 0].reset_index(drop=True)

    if data.empty:
        raise ValueError("merge 후 리뷰 0건")

    return data, target_cols


def print_data_summary(data: pd.DataFrame) -> None:
    n_reviews = len(data)
    n_courses = data["course_key"].nunique()
    n_profs = data["professor"].fillna("").nunique()
    rpc = data.groupby("course_key").size()
    print(f"  리뷰 {n_reviews:,}건 | 과목 {n_courses} | 교수 {n_profs}")
    print(f"  과목당 리뷰: min={rpc.min()} med={rpc.median():.0f} "
          f"max={rpc.max()} mean={rpc.mean():.1f}")


# ════════════════════════════════════════════════════════════════
# 4. Metrics
# ════════════════════════════════════════════════════════════════

def _safe_pearson(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(pearsonr(a, b).statistic)


@dataclass
class FoldMetrics:
    fold: int
    target: str
    model: str
    mae: float
    rmse: float
    pearson_r: float


def _eval_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    fold: int,
    model_name: str,
) -> list[FoldMetrics]:
    rows = []
    for i, t in enumerate(targets):
        rows.append(FoldMetrics(
            fold=fold, target=t, model=model_name,
            mae=mean_absolute_error(y_true[:, i], y_pred[:, i]),
            rmse=math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            pearson_r=_safe_pearson(y_true[:, i], y_pred[:, i]),
        ))
    rows.append(FoldMetrics(
        fold=fold, target="average", model=model_name,
        mae=mean_absolute_error(y_true.ravel(), y_pred.ravel()),
        rmse=math.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())),
        pearson_r=_safe_pearson(y_true.ravel(), y_pred.ravel()),
    ))
    return rows


def metrics_to_df(rows: list[FoldMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "target"], as_index=False)
        .agg(
            MAE_mean=("mae", "mean"),
            MAE_std=("mae", lambda x: x.std(ddof=0)),
            RMSE_mean=("rmse", "mean"),
            RMSE_std=("rmse", lambda x: x.std(ddof=0)),
            Pearson_mean=("pearson_r", "mean"),
            Pearson_std=("pearson_r", lambda x: x.std(ddof=0)),
        )
    )


# ════════════════════════════════════════════════════════════════
# 5. Single-fold training
# ════════════════════════════════════════════════════════════════

def _build_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_cols: list[str],
    hparams: dict,
    device: torch.device,
):
    vec = TfidfVectorizer(
        max_features=hparams["max_features"],
        ngram_range=hparams["ngram_range"],
        min_df=hparams["min_df"],
    )
    X_tr = torch.tensor(
        vec.fit_transform(train_df["raw_review_text"]).toarray(),
        dtype=torch.float32, device=device,
    )
    X_va = torch.tensor(
        vec.transform(val_df["raw_review_text"]).toarray(),
        dtype=torch.float32, device=device,
    )

    def _group(frame):
        keys = sorted(frame["course_key"].unique())
        k2i = {k: i for i, k in enumerate(keys)}
        idx = torch.tensor(
            [k2i[k] for k in frame["course_key"]],
            dtype=torch.long, device=device,
        )
        tgt = (
            frame[["course_key", *target_cols]]
            .drop_duplicates("course_key")
            .sort_values("course_key")
        )
        y = torch.tensor(tgt[target_cols].values, dtype=torch.float32, device=device)
        return keys, idx, y

    tr_keys, tr_idx, y_tr = _group(train_df)
    va_keys, va_idx, y_va = _group(val_df)
    return X_tr, X_va, tr_keys, tr_idx, y_tr, va_keys, va_idx, y_va


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_cols: list[str],
    n_targets: int,
    hparams: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """과목 단위 (y_true, y_pred) 반환."""
    torch.manual_seed(seed)

    (X_tr, X_va,
     tr_keys, tr_idx, y_tr,
     va_keys, va_idx, y_va) = _build_tensors(
        train_df, val_df, target_cols, hparams, device,
    )

    model = SmallScoreModel(
        input_dim=X_tr.shape[1],
        output_dim=n_targets,
        hidden_dim=hparams["hidden_dim"],
        dropout=hparams["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=FIXED_TRAINING["weight_decay"],
    )

    n_tr = len(tr_keys)
    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(1, FIXED_TRAINING["epochs"] + 1):
        model.train()
        opt.zero_grad()

        pred_avg = group_average(model(X_tr), tr_idx, n_tr)
        per_course = ((pred_avg - y_tr) ** 2).mean(dim=1)
        counts = torch.bincount(tr_idx, minlength=n_tr).float().to(device)
        loss = (per_course * counts).sum() / counts.sum()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_avg = group_average(model(X_va), va_idx, len(va_keys))
            val_mse = float(nn.MSELoss()(va_avg, y_va))

        if val_mse < best_val - FIXED_TRAINING["min_delta"]:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= FIXED_TRAINING["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        va_pred = group_average(model(X_va), va_idx, len(va_keys))

    return y_va.cpu().numpy(), va_pred.cpu().numpy()


# ════════════════════════════════════════════════════════════════
# 6. CV experiment — 한 축, 한 설정
# ════════════════════════════════════════════════════════════════

def _split_folds(
    data: pd.DataFrame,
    axis: SplitAxis,
    n_splits: int,
):
    course_frame = (
        data[["course_key", "professor"]]
        .drop_duplicates("course_key")
        .reset_index(drop=True)
    )
    keys_arr = course_frame["course_key"].values

    if axis == "course":
        groups = keys_arr
    else:
        groups = course_frame["professor"].fillna("").str.strip().values

    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr_i, va_i in gkf.split(keys_arr, groups=groups):
        tr_set = set(keys_arr[tr_i])
        va_set = set(keys_arr[va_i])
        folds.append((
            data[data["course_key"].isin(tr_set)].reset_index(drop=True),
            data[data["course_key"].isin(va_set)].reset_index(drop=True),
        ))
    return folds


def run_cv(
    data: pd.DataFrame,
    target_cols: list[str],
    targets: list[str],
    hparams: dict,
    axis: SplitAxis,
    n_splits: int,
    device: torch.device,
    seed: int,
    label: str = "tfidf_mlp",
) -> pd.DataFrame:
    folds = _split_folds(data, axis, n_splits)
    all_metrics: list[FoldMetrics] = []
    t0 = time.time()

    for i, (tr, va) in enumerate(folds, 1):
        ft0 = time.time()
        y_true, y_pred = train_one_fold(
            tr, va, target_cols, len(targets), hparams, device, seed + i,
        )
        elapsed = time.time() - ft0
        total = time.time() - t0
        eta = total / i * (n_splits - i)
        print(f"      fold {i}/{n_splits} done {elapsed:.0f}s "
              f"(elapsed {total:.0f}s, ETA {eta:.0f}s)", flush=True)

        baseline = np.tile(y_true.mean(axis=0), (len(y_true), 1))
        all_metrics.extend(_eval_targets(y_true, baseline, targets, i, "baseline"))
        all_metrics.extend(_eval_targets(y_true, y_pred, targets, i, label))

    return metrics_to_df(all_metrics)


# ════════════════════════════════════════════════════════════════
# 7. 하이퍼파라미터 서치
# ════════════════════════════════════════════════════════════════

def run_search(
    data: pd.DataFrame,
    target_cols: list[str],
    targets: list[str],
    axis: SplitAxis,
    n_splits: int,
    n_iter: int,
    device: torch.device,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    candidates = list(ParameterSampler(
        SEARCH_SPACE, n_iter=n_iter, random_state=seed,
    ))

    records = []
    best_mae, best_id = float("inf"), 0
    search_t0 = time.time()

    for pid, params in enumerate(candidates, 1):
        hp = {**UPSTREAM_DEFAULTS, **params}
        print(f"    [{pid:02d}/{n_iter}] {params}", flush=True)

        fold_df = run_cv(
            data, target_cols, targets, hp, axis, n_splits,
            device, seed, label=f"search_{pid:02d}",
        )

        avg = fold_df[
            (fold_df["model"] == f"search_{pid:02d}")
            & (fold_df["target"] == "average")
        ]
        mae_mean = avg["mae"].mean()
        mae_std = avg["mae"].std(ddof=0)

        records.append({
            "param_id": pid,
            **{k: str(v) for k, v in params.items()},
            "cv_MAE_mean": mae_mean,
            "cv_MAE_std": mae_std,
        })
        s_elapsed = time.time() - search_t0
        s_eta = s_elapsed / pid * (n_iter - pid)
        print(f"         MAE={mae_mean:.4f} ± {mae_std:.4f}  "
              f"[search {s_elapsed:.0f}s elapsed, ETA {s_eta:.0f}s]",
              flush=True)

        if mae_mean < best_mae:
            best_mae = mae_mean
            best_id = pid

    search_df = pd.DataFrame(records).sort_values("cv_MAE_mean").reset_index(drop=True)
    best_params = {**UPSTREAM_DEFAULTS, **candidates[best_id - 1]}

    best_fold_df = run_cv(
        data, target_cols, targets, best_params, axis, n_splits,
        device, seed, label="best_search",
    )

    return best_params, search_df, best_fold_df


# ════════════════════════════════════════════════════════════════
# 8. 출력
# ════════════════════════════════════════════════════════════════

def _ms(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.3f} ± {std:.3f}" if not pd.isna(std) else f"{mean:.3f}"


def write_slide_table(
    baseline_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    best_params: dict,
    axis: SplitAxis,
    n_splits: int,
    out: Path,
) -> str:
    target_order = TARGETS + ["average"]
    rows = []
    for t in target_order:
        bl = baseline_summary[
            (baseline_summary["model"] == "baseline") & (baseline_summary["target"] == t)
        ].iloc[0]
        rows.append({
            "Target": t,
            "Model": "Global Mean",
            "MAE": _ms(bl["MAE_mean"], bl["MAE_std"]),
            "RMSE": _ms(bl["RMSE_mean"], bl["RMSE_std"]),
            "Pearson r": _ms(bl["Pearson_mean"], bl["Pearson_std"]),
        })

        for label, summary in [
            ("Baseline HP", baseline_summary),
            ("Best Search", best_summary),
        ]:
            model_key = "tfidf_mlp" if label == "Baseline HP" else "best_search"
            r = summary[
                (summary["model"] == model_key) & (summary["target"] == t)
            ].iloc[0]
            rows.append({
                "Target": t,
                "Model": label,
                "MAE": _ms(r["MAE_mean"], r["MAE_std"]),
                "RMSE": _ms(r["RMSE_mean"], r["RMSE_std"]),
                "Pearson r": _ms(r["Pearson_mean"], r["Pearson_std"]),
            })

    table_df = pd.DataFrame(rows)
    axis_label = "leave-course-out" if axis == "course" else "leave-professor-out"

    bl_avg = baseline_summary[
        (baseline_summary["model"] == "tfidf_mlp")
        & (baseline_summary["target"] == "average")
    ]["MAE_mean"].iloc[0]
    best_avg = best_summary[
        (best_summary["model"] == "best_search")
        & (best_summary["target"] == "average")
    ]["MAE_mean"].iloc[0]
    impr = (bl_avg - best_avg) / bl_avg * 100

    md = (
        f"## MIL CV — {axis_label}\n\n"
        f"- Protocol: {n_splits}-fold GroupKFold ({axis_label})\n"
        f"- Baseline HP: `{UPSTREAM_DEFAULTS}`\n"
        f"- Best search: `{best_params}`\n"
        f"- MAE 개선 (baseline HP → best): **{impr:+.1f}%**\n\n"
        f"{table_df.to_markdown(index=False)}\n"
    )

    (out / "slide_table.md").write_text(md, encoding="utf-8")
    return md


def write_combined_table(results: dict, out: Path) -> None:
    target_order = TARGETS + ["average"]
    rows = []
    for t in target_order:
        row = {"Target": t}
        for axis in ["course", "professor"]:
            bl = results[axis]["baseline_summary"]
            bs = results[axis]["best_summary"]
            bl_r = bl[(bl["model"] == "tfidf_mlp") & (bl["target"] == t)].iloc[0]
            bs_r = bs[(bs["model"] == "best_search") & (bs["target"] == t)].iloc[0]
            al = "CO" if axis == "course" else "PO"
            row[f"{al} Baseline MAE"] = _ms(bl_r["MAE_mean"], bl_r["MAE_std"])
            row[f"{al} Best MAE"] = _ms(bs_r["MAE_mean"], bs_r["MAE_std"])
            row[f"{al} Best r"] = _ms(bs_r["Pearson_mean"], bs_r["Pearson_std"])
        rows.append(row)

    md = (
        "## MIL CV — 두 축 비교\n\n"
        "CO = leave-course-out, PO = leave-professor-out\n\n"
        f"{pd.DataFrame(rows).to_markdown(index=False)}\n"
    )
    (out / "combined_slide_table.md").write_text(md, encoding="utf-8")
    print(md)


# ════════════════════════════════════════════════════════════════
# 9. Main
# ════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="MIL CV Experiment")
    p.add_argument("--raw", default=os.environ.get(
        "RAW_PATH",
        "../../references/OptiLife/data/csv/raw_everytime_reviews.csv",
    ))
    p.add_argument("--courses", default=os.environ.get(
        "COURSES_PATH",
        "../../references/OptiLife/data/csv/courses.csv",
    ))
    p.add_argument("--output-dir", default="mil_cv_outputs")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--axes", default="course,professor",
                   help="comma-separated axes to run (course,professor)")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("MIL CV Experiment")
    print("=" * 60)

    data, target_cols = load_data(args.raw, args.courses, TARGETS)
    print_data_summary(data)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    axes = [a.strip() for a in args.axes.split(",")]

    for axis in axes:
        axis_dir = out / f"leave_{axis}_out"
        axis_dir.mkdir(parents=True, exist_ok=True)
        axis_label = "leave-course-out" if axis == "course" else "leave-professor-out"

        print(f"\n{'=' * 60}")
        print(f"  {axis_label}")
        print(f"{'=' * 60}")

        # ── baseline ──
        print(f"\n  [baseline] upstream 고정 HP")
        bl_df = run_cv(
            data, target_cols, TARGETS, UPSTREAM_DEFAULTS,
            axis, args.n_splits, device, args.seed,
        )
        bl_summary = summarize(bl_df)
        bl_df.to_csv(axis_dir / "baseline_fold_metrics.csv",
                      index=False, encoding="utf-8-sig")
        bl_summary.to_csv(axis_dir / "baseline_summary.csv",
                          index=False, encoding="utf-8-sig")

        bl_model = bl_summary[
            (bl_summary["model"] == "tfidf_mlp")
            & (bl_summary["target"] == "average")
        ].iloc[0]
        print(f"  baseline MAE={bl_model['MAE_mean']:.4f} "
              f"± {bl_model['MAE_std']:.4f}")

        # ── search ──
        print(f"\n  [search] {args.n_iter}-iter 랜덤 서치")
        best_params, search_df, best_df = run_search(
            data, target_cols, TARGETS,
            axis, args.n_splits, args.n_iter, device, args.seed,
        )
        best_summary = summarize(best_df)

        search_df.to_csv(axis_dir / "hparam_search_summary.csv",
                         index=False, encoding="utf-8-sig")
        best_df.to_csv(axis_dir / "best_fold_metrics.csv",
                       index=False, encoding="utf-8-sig")
        best_summary.to_csv(axis_dir / "best_summary.csv",
                            index=False, encoding="utf-8-sig")

        bp_safe = {k: list(v) if isinstance(v, tuple) else v
                   for k, v in best_params.items()}
        (axis_dir / "best_params.json").write_text(
            json.dumps(bp_safe, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        bs_model = best_summary[
            (best_summary["model"] == "best_search")
            & (best_summary["target"] == "average")
        ].iloc[0]
        print(f"\n  best MAE={bs_model['MAE_mean']:.4f} "
              f"± {bs_model['MAE_std']:.4f}")
        print(f"  best params: {best_params}")

        # ── slide table ──
        md = write_slide_table(
            bl_summary, best_summary, best_params,
            axis, args.n_splits, axis_dir,
        )
        print(f"\n{md}")

        results[axis] = dict(
            baseline_summary=bl_summary,
            best_summary=best_summary,
            best_params=best_params,
        )

    # ── combined ──
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("  Combined")
        print("=" * 60)
        write_combined_table(results, out)

    print(f"\n출력: {out.resolve()}")


if __name__ == "__main__":
    main()
