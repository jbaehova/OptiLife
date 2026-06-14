# Cell 3: 모델 정의
import math, time, json, gc
from pathlib import Path
from dataclasses import dataclass, asdict

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

class ReviewDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        if self.labels is None: return self.texts[idx]
        return self.texts[idx], self.labels[idx]

class BertRegressor(nn.Module):
    def __init__(self, model_name, output_dim, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, output_dim)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        out = self.bert(**kw)
        pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        return 1.0 + 4.0 * torch.sigmoid(self.head(self.dropout(pooled)))

def make_collate(tokenizer, max_len, has_labels):
    def fn(batch):
        if has_labels:
            texts, labels = zip(*batch)
            enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            return enc, torch.tensor(np.asarray(labels, dtype=np.float32))
        enc = tokenizer(list(batch), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc
    return fn

def safe_pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8: return np.nan
    return float(pearsonr(a, b).statistic)

print("Model definitions loaded.")



# Cell 4: 학습 + CV 함수

def train_bert_fold(train_df, val_df, hparams, device, fold_name="Fold"):
    """한 fold 학습. 과목 단위 (y_true, y_pred) 반환."""
    t0 = time.time()
    model_name = hparams["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_labels = train_df[TARGET_COLS].values.astype(np.float32)
    val_labels = val_df[TARGET_COLS].values.astype(np.float32)

    train_loader = DataLoader(
        ReviewDataset(train_df["raw_review_text"].values, train_labels),
        batch_size=hparams["batch_size"], shuffle=True,
        collate_fn=make_collate(tokenizer, hparams["max_length"], True),
    )
    val_loader = DataLoader(
        ReviewDataset(val_df["raw_review_text"].values, val_labels),
        batch_size=hparams["batch_size"], shuffle=False,
        collate_fn=make_collate(tokenizer, hparams["max_length"], True),
    )

    model = BertRegressor(model_name, len(TARGETS), hparams["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * hparams["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(1, hparams["epochs"] + 1):
        model.train()
        train_loss, n = 0.0, 0
        for enc, labels in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(**enc)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach()) * labels.shape[0]
            n += labels.shape[0]

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for enc, labels in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                val_loss += float(criterion(model(**enc), labels)) * labels.shape[0]
                vn += labels.shape[0]

        t_mse, v_mse = train_loss/n, val_loss/vn
        print(f"  {fold_name} ep{epoch}/{hparams['epochs']} train={t_mse:.4f} val={v_mse:.4f} ({time.time()-t0:.0f}s)")
        if v_mse < best_val:
            best_val = v_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)

    # 과목 단위 예측
    model.eval()
    preds = []
    with torch.no_grad():
        for enc, _ in val_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            preds.append(model(**enc).cpu().numpy())
    review_preds = np.clip(np.vstack(preds), 1.0, 5.0)

    # 과목 단위 집계
    pred_df = pd.DataFrame({"course_key": val_df["course_key"].values})
    for i, t in enumerate(TARGETS):
        pred_df[f"pred_{t}"] = review_preds[:, i]
        pred_df[f"true_{t}"] = val_df[TARGET_COLS[i]].values
    course_df = pred_df.groupby("course_key").agg(
        {**{f"pred_{t}": "mean" for t in TARGETS}, **{f"true_{t}": "first" for t in TARGETS}}
    ).reset_index()

    y_true = course_df[[f"true_{t}" for t in TARGETS]].values
    y_pred = course_df[[f"pred_{t}" for t in TARGETS]].values

    del model, tokenizer, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return y_true, y_pred, time.time() - t0


def run_bert_cv(data, hparams, n_splits=5, split_axis="course"):
    """두 축 GroupKFold CV."""
    course_frame = data[["course_key", "professor"]].drop_duplicates("course_key").reset_index(drop=True)
    keys = course_frame["course_key"].values
    groups = keys if split_axis == "course" else course_frame["professor"].fillna("").str.strip().values

    gkf = GroupKFold(n_splits=n_splits)
    all_metrics = []
    cv_t0 = time.time()

    for fold, (tr_i, va_i) in enumerate(gkf.split(keys, groups=groups), 1):
        tr_set, va_set = set(keys[tr_i]), set(keys[va_i])
        tr_df = data[data["course_key"].isin(tr_set)].reset_index(drop=True)
        va_df = data[data["course_key"].isin(va_set)].reset_index(drop=True)
        print(f"\nFold {fold}/{n_splits}: train {len(tr_df)} reviews ({tr_df['course_key'].nunique()} courses), val {len(va_df)} ({va_df['course_key'].nunique()})")

        y_true, y_pred, elapsed = train_bert_fold(tr_df, va_df, hparams, DEVICE, f"F{fold}")
        baseline = np.tile(y_true.mean(axis=0), (len(y_true), 1))

        for label, yp in [("baseline", baseline), ("bert", y_pred)]:
            for i, t in enumerate(TARGETS):
                all_metrics.append({"fold": fold, "target": t, "model": label,
                    "mae": mean_absolute_error(y_true[:,i], yp[:,i]),
                    "rmse": math.sqrt(mean_squared_error(y_true[:,i], yp[:,i])),
                    "pearson_r": safe_pearson(y_true[:,i], yp[:,i])})
            all_metrics.append({"fold": fold, "target": "average", "model": label,
                "mae": mean_absolute_error(y_true.ravel(), yp.ravel()),
                "rmse": math.sqrt(mean_squared_error(y_true.ravel(), yp.ravel())),
                "pearson_r": safe_pearson(y_true.ravel(), yp.ravel())})

        total = time.time() - cv_t0
        eta = total / fold * (n_splits - fold)
        print(f"  fold {fold} done {elapsed:.0f}s (total {total:.0f}s, ETA {eta:.0f}s)")

    return pd.DataFrame(all_metrics)

print("CV functions loaded.")



# Cell 5: Baseline — 5-fold GroupKFold × 3 epochs (upstream 설정)
FAST_HP_CO = {
    "model_name": "klue/bert-base",
    "epochs": 3,
    "batch_size": 128,
    "max_length": 128,
    "lr": 2e-5,
    "dropout": 0.1,
}

print("=" * 60)
print("BASELINE: leave-course-out (batch=128)")
print("=" * 60)
bl_co = run_bert_cv(data, FAST_HP_CO, n_splits=5, split_axis="course")

bl_co_summary = bl_co.groupby(["model", "target"], as_index=False).agg(
    MAE_mean=("mae", "mean"), MAE_std=("mae", lambda x: x.std(ddof=0)),
    RMSE_mean=("rmse", "mean"), Pearson_mean=("pearson_r", "mean"),
    Pearson_std=("pearson_r", lambda x: x.std(ddof=0)),
)
print("\n" + bl_co_summary.to_string(index=False))



import logging; logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

FAST_HP = {
    "model_name": "klue/bert-base",
    "epochs": 3,
    "batch_size": 128,
    "max_length": 128,
    "lr": 2e-5,
    "dropout": 0.1,
}

print("=" * 60)
print("BASELINE: leave-professor-out (batch=128)")
print("=" * 60)
bl_po = run_bert_cv(data, FAST_HP, n_splits=5, split_axis="professor")

bl_po_summary = bl_po.groupby(["model", "target"], as_index=False).agg(
    MAE_mean=("mae", "mean"), MAE_std=("mae", lambda x: x.std(ddof=0)),
    RMSE_mean=("rmse", "mean"), Pearson_mean=("pearson_r", "mean"),
    Pearson_std=("pearson_r", lambda x: x.std(ddof=0)),
)
print("\n" + bl_po_summary.to_string(index=False))



from pathlib import Path
OUT = Path("/content/bert_cv_results")
OUT.mkdir(exist_ok=True)

# fold 메트릭 저장
bl_co.to_csv(OUT / "bert_co_fold_metrics.csv", index=False, encoding="utf-8-sig")
bl_co_summary.to_csv(OUT / "bert_co_summary.csv", index=False, encoding="utf-8-sig")
bl_po.to_csv(OUT / "bert_po_fold_metrics.csv", index=False, encoding="utf-8-sig")
bl_po_summary.to_csv(OUT / "bert_po_summary.csv", index=False, encoding="utf-8-sig")

# 비교표
tfidf = {"workload": (0.436, 0.684, 0.439, 0.667),
         "teamwork": (0.657, 0.736, 0.662, 0.745),
         "grading": (0.404, 0.457, 0.413, 0.424),
         "average": (0.499, 0.721, 0.505, 0.720)}
tmap = {"workload_label":"workload","teamwork_load_label":"teamwork",
        "grading_strictness_label":"grading","average":"average"}
ms = lambda m,s: f"{m:.3f} ± {s:.3f}"

rows = []
for t in ["workload_label","teamwork_load_label","grading_strictness_label","average"]:
    s = tmap[t]
    co = bl_co_summary[(bl_co_summary["model"]=="bert")&(bl_co_summary["target"]==t)].iloc[0]
    po = bl_po_summary[(bl_po_summary["model"]=="bert")&(bl_po_summary["target"]==t)].iloc[0]
    tf = tfidf[s]
    rows.append({
        "Target": s,
        "TF-IDF CO MAE": f"{tf[0]:.3f}", "BERT CO MAE": ms(co["MAE_mean"],co["MAE_std"]),
        "TF-IDF CO r": f"{tf[1]:.3f}", "BERT CO r": ms(co["Pearson_mean"],co["Pearson_std"]),
        "TF-IDF PO MAE": f"{tf[2]:.3f}", "BERT PO MAE": ms(po["MAE_mean"],po["MAE_std"]),
        "TF-IDF PO r": f"{tf[3]:.3f}", "BERT PO r": ms(po["Pearson_mean"],po["Pearson_std"]),
    })

compare = pd.DataFrame(rows)
md = "## TF-IDF+MLP vs BERT — CO + PO 비교\n\n" + compare.to_markdown(index=False)
(OUT / "full_comparison.md").write_text(md, encoding="utf-8")
compare.to_csv(OUT / "full_comparison.csv", index=False, encoding="utf-8-sig")
print(md)



# Cell 7a: 서치 전반 (1-4)

from sklearn.model_selection import ParameterSampler

SEARCH_SPACE = {
    "epochs": [2, 3, 4],
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [64, 128],
    "max_length": [128, 256],
}

ALL_CANDIDATES = list(ParameterSampler(SEARCH_SPACE, n_iter=8, random_state=42))
search_results = []
search_t0 = time.time()

for pid, params in enumerate(ALL_CANDIDATES[:4], 1):
    hp = {"model_name": "klue/bert-base", **params}
    print(f"\n{'='*60}\n[{pid}/4] {params}\n{'='*60}")
    fold_df = run_bert_cv(data, hp, n_splits=5, split_axis="course")
    avg = fold_df[(fold_df["model"]=="bert") & (fold_df["target"]=="average")]
    mae_mean, mae_std = avg["mae"].mean(), avg["mae"].std(ddof=0)
    search_results.append({"param_id": pid, **{k: str(v) for k,v in params.items()},
                           "cv_MAE_mean": mae_mean, "cv_MAE_std": mae_std})
    elapsed = time.time() - search_t0
    print(f"  MAE={mae_mean:.4f} ± {mae_std:.4f} [{elapsed:.0f}s elapsed, ETA {elapsed/pid*(4-pid):.0f}s]")

search_df = pd.DataFrame(search_results).sort_values("cv_MAE_mean").reset_index(drop=True)
print("\n=== 전반 결과 ===")
print(search_df.to_string(index=False))



# Cell 7b: 서치 후반 (5-8)

for pid_offset, params in enumerate(ALL_CANDIDATES[4:8], 5):
    hp = {"model_name": "klue/bert-base", **params}
    print(f"\n{'='*60}\n[{pid_offset}/8] {params}\n{'='*60}")
    fold_df = run_bert_cv(data, hp, n_splits=5, split_axis="course")
    avg = fold_df[(fold_df["model"]=="bert") & (fold_df["target"]=="average")]
    mae_mean, mae_std = avg["mae"].mean(), avg["mae"].std(ddof=0)
    search_results.append({"param_id": pid_offset, **{k: str(v) for k,v in params.items()},
                           "cv_MAE_mean": mae_mean, "cv_MAE_std": mae_std})
    elapsed = time.time() - search_t0
    remain = 8 - pid_offset
    print(f"  MAE={mae_mean:.4f} ± {mae_std:.4f} [{elapsed:.0f}s elapsed, ETA {elapsed/(pid_offset-4)*remain:.0f}s]")

search_df = pd.DataFrame(search_results).sort_values("cv_MAE_mean").reset_index(drop=True)
print("\n=== 전체 결과 ===")
print(search_df.to_string(index=False))



from sklearn.model_selection import ParameterSampler

SEARCH_SPACE = {
    "epochs": [2, 3, 4],
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [64, 128],
    "max_length": [128, 256],
}

N_ITER = 8
candidates = list(ParameterSampler(SEARCH_SPACE, n_iter=N_ITER, random_state=42))

search_results = []
search_t0 = time.time()

for pid, params in enumerate(candidates, 1):
    hp = {"model_name": "klue/bert-base", **params}
    print(f"\n{'='*60}")
    print(f"[{pid}/{N_ITER}] {params}")
    print(f"{'='*60}")

    fold_df = run_bert_cv(data, hp, n_splits=5, split_axis="course")
    avg = fold_df[(fold_df["model"]=="bert") & (fold_df["target"]=="average")]
    mae_mean = avg["mae"].mean()
    mae_std = avg["mae"].std(ddof=0)

    search_results.append({"param_id": pid, **{k: str(v) for k,v in params.items()},
                           "cv_MAE_mean": mae_mean, "cv_MAE_std": mae_std})

    s_elapsed = time.time() - search_t0
    s_eta = s_elapsed / pid * (N_ITER - pid)
    print(f"  MAE={mae_mean:.4f} ± {mae_std:.4f} [elapsed {s_elapsed:.0f}s, ETA {s_eta:.0f}s]")

search_df = pd.DataFrame(search_results).sort_values("cv_MAE_mean").reset_index(drop=True)
print("\n" + "=" * 60)
print("SEARCH RESULTS (sorted by MAE)")
print("=" * 60)
print(search_df.to_string(index=False))



# Cell 8: 최종 비교표
def ms(mean, std):
    if pd.isna(mean): return "-"
    return f"{mean:.3f} ± {std:.3f}" if not pd.isna(std) else f"{mean:.3f}"

# TF-IDF+MLP baseline (이전 실험 결과 수동 입력)
tfidf_results = {
    "workload": {"mae": 0.436, "r": 0.684},
    "teamwork": {"mae": 0.657, "r": 0.736},
    "grading": {"mae": 0.404, "r": 0.457},
    "average": {"mae": 0.499, "r": 0.721},
}

target_order = ["workload_label", "teamwork_load_label", "grading_strictness_label", "average"]
short = {"workload_label": "workload", "teamwork_load_label": "teamwork",
         "grading_strictness_label": "grading", "average": "average"}

rows = []
for t in target_order:
    bl = bl_co_summary[(bl_co_summary["model"]=="bert") & (bl_co_summary["target"]==t)].iloc[0]
    tf = tfidf_results[short[t]]
    rows.append({
        "Target": short[t],
        "TF-IDF+MLP MAE": f"{tf['mae']:.3f}",
        "TF-IDF+MLP r": f"{tf['r']:.3f}",
        "BERT MAE": ms(bl["MAE_mean"], bl["MAE_std"]),
        "BERT r": ms(bl["Pearson_mean"], bl["Pearson_std"]),
    })

compare = pd.DataFrame(rows)
print("\n## TF-IDF+MLP vs BERT (leave-course-out)\n")
print(compare.to_markdown(index=False))

if len(search_df) > 0:
    best = search_df.iloc[0]
    print(f"\n## Best BERT search: MAE={best['cv_MAE_mean']:.4f} ± {best['cv_MAE_std']:.4f}")
    print(f"Params: {dict(search_df.iloc[0].drop(['param_id','cv_MAE_mean','cv_MAE_std']))}")


