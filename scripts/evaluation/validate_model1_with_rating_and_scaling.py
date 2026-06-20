#rating에 대해서 잘 되는지 실효성 검사

import argparseimport mathimport osfrom pathlib import Path

import numpy as npimport pandas as pdfrom scipy.stats import pearsonrfrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.linear_model import Ridgefrom sklearn.metrics import mean_absolute_error, mean_squared_errorfrom sklearn.model_selection import train_test_splitfrom sklearn.neural_network import MLPRegressorfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScaler

============================================================

Utilities

============================================================

def clean_text(x):return "" if pd.isna(x) else str(x).strip()

def make_course_key(df):return (df["course_name"].astype(str).str.strip()+ "__"+ df["professor"].fillna("").astype(str).str.strip())

def safe_pearson(true, pred):true = np.asarray(true)pred = np.asarray(pred)

if len(true) < 2:
    return np.nan
if np.std(true) < 1e-8 or np.std(pred) < 1e-8:
    return np.nan

return pearsonr(true, pred).statistic

def evaluate_predictions(y_true, y_pred, label):y_true = np.asarray(y_true)y_pred = np.asarray(y_pred)

return {
    "model": label,
    "MAE": mean_absolute_error(y_true, y_pred),
    "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
    "Pearson_r": safe_pearson(y_true, y_pred),
}

def add_scaled_to_course_mean(frame, pred_col, baseline_col="course_mean_rating"):out = frame.copy()

stats = (
    out.groupby("course_key")
    .agg(
        pred_mean=(pred_col, "mean"),
        true_mean=(baseline_col, "first"),
    )
    .reset_index()
)

scale_col = f"{pred_col}_scale"
scaled_col = f"{pred_col}_scaled_to_course_mean"

stats[scale_col] = stats["true_mean"] / (stats["pred_mean"] + 1e-8)

out = out.merge(
    stats[["course_key", scale_col]],
    on="course_key",
    how="left",
)

out[scaled_col] = (out[pred_col] * out[scale_col]).clip(1.0, 5.0)

return out, scaled_col

def load_rating_data(raw_path):raw = pd.read_csv(raw_path)

required = ["course_name", "professor", "rating", "raw_review_text"]
missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"raw file missing columns: {missing}")

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
    raise ValueError("No usable rating reviews.")

# This is intentionally computed from all reviews.
# The experiment assumes course mean rating is known and tests whether text adds review-level variation.
raw["course_mean_rating"] = raw.groupby("course_key")["rating"].transform("mean")

return raw

============================================================

Model 1: TF-IDF + Ridge

============================================================

def run_tfidf_ridge(train_df, val_df, args):model = Pipeline([("tfidf", TfidfVectorizer(max_features=args.max_features,ngram_range=(1, 2),min_df=args.min_df,)),("ridge", Ridge(alpha=args.ridge_alpha)),])

model.fit(
    train_df["raw_review_text"].values,
    train_df["course_mean_rating"].values,
)

pred = model.predict(val_df["raw_review_text"].values)
return np.clip(pred, 1.0, 5.0)

============================================================

Model 2: TF-IDF + MLP

============================================================

def run_tfidf_mlp(train_df, val_df, args):model = Pipeline([("tfidf", TfidfVectorizer(max_features=args.max_features,ngram_range=(1, 2),min_df=args.min_df,)),("mlp", MLPRegressor(hidden_layer_sizes=(args.mlp_hidden_dim,),activation="relu",alpha=args.mlp_alpha,learning_rate_init=args.mlp_lr,max_iter=args.mlp_max_iter,early_stopping=True,n_iter_no_change=args.mlp_patience,random_state=args.seed,verbose=False,)),])

model.fit(
    train_df["raw_review_text"].values,
    train_df["course_mean_rating"].values,
)

pred = model.predict(val_df["raw_review_text"].values)
return np.clip(pred, 1.0, 5.0)

============================================================

Model 3: SBERT embedding + Ridge

============================================================

def run_sbert_ridge(train_df, val_df, args):try:from sentence_transformers import SentenceTransformerexcept ImportError as e:raise ImportError("sentence-transformers is required for sbert_ridge.\n""Install: pip install sentence-transformers") from e

embedder = SentenceTransformer(args.sbert_model)

X_train = embedder.encode(
    train_df["raw_review_text"].tolist(),
    batch_size=args.sbert_batch_size,
    show_progress_bar=True,
    normalize_embeddings=True,
)
X_val = embedder.encode(
    val_df["raw_review_text"].tolist(),
    batch_size=args.sbert_batch_size,
    show_progress_bar=True,
    normalize_embeddings=True,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=args.ridge_alpha)),
])

model.fit(X_train, train_df["course_mean_rating"].values)
pred = model.predict(X_val)
return np.clip(pred, 1.0, 5.0)

============================================================

Model 4: BERT fine-tuning

============================================================

def run_bert_finetune(train_df, val_df, args):try:import torchimport torch.nn as nnfrom torch.utils.data import Dataset, DataLoaderfrom transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmupexcept ImportError as e:raise ImportError("torch and transformers are required for bert_finetune.\n""Install: pip install torch transformers accelerate") from e

device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

class RatingDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=args.bert_max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    return enc, labels

train_ds = RatingDataset(
    train_df["raw_review_text"].values,
    train_df["course_mean_rating"].values,
)
val_ds = RatingDataset(
    val_df["raw_review_text"].values,
    val_df["course_mean_rating"].values,
)

train_loader = DataLoader(
    train_ds,
    batch_size=args.bert_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.bert_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

class BertRegressor(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        raw = self.head(self.dropout(pooled)).squeeze(-1)
        return 1.0 + 4.0 * torch.sigmoid(raw)

model = BertRegressor(args.bert_model, dropout=args.bert_dropout).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.bert_lr,
    weight_decay=args.bert_weight_decay,
)

total_steps = len(train_loader) * args.bert_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * args.bert_warmup_ratio),
    num_training_steps=total_steps,
)

criterion = nn.MSELoss()
best_val = float("inf")
best_state = None

for epoch in range(1, args.bert_epochs + 1):
    model.train()
    train_losses = []

    for enc, labels in train_loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(**enc)
        loss = criterion(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.bert_grad_clip)
        optimizer.step()
        scheduler.step()

        train_losses.append(float(loss.detach().cpu()))

    model.eval()
    val_losses = []
    with torch.no_grad():
        for enc, labels in val_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            pred = model(**enc)
            loss = criterion(pred, labels)
            val_losses.append(float(loss.detach().cpu()))

    train_mse = float(np.mean(train_losses))
    val_mse = float(np.mean(val_losses))

    print(
        f"BERT Epoch {epoch:02d} | "
        f"train_mse={train_mse:.4f} | val_course_mean_mse={val_mse:.4f}"
    )

    if val_mse < best_val:
        best_val = val_mse
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
preds = []
with torch.no_grad():
    for enc, _ in val_loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        pred = model(**enc)
        preds.append(pred.detach().cpu().numpy())

pred = np.concatenate(preds)
return np.clip(pred, 1.0, 5.0)

============================================================

Experiment runner

============================================================

def run_experiment(args):output_dir = Path(args.output_dir)output_dir.mkdir(parents=True, exist_ok=True)

raw = load_rating_data(args.raw)

train_df, val_df = train_test_split(
    raw,
    test_size=args.val_size,
    random_state=args.seed,
    shuffle=True,
)

print("rating reviews:", len(raw))
print("course groups:", raw["course_key"].nunique())
print("train reviews:", len(train_df))
print("validation reviews:", len(val_df))
print("models:", args.models)

val_out_cols = [
    "course_key",
    "course_name",
    "professor",
    "rating",
    "course_mean_rating",
    "raw_review_text",
]
if "semester" in val_df.columns:
    val_out_cols.insert(3, "semester")

val_out = val_df[val_out_cols].copy()
val_out = val_out.rename(columns={
    "rating": "true_rating",
    "course_mean_rating": "baseline_course_mean_rating",
})

y_true = val_out["true_rating"].values
baseline = val_out["baseline_course_mean_rating"].values

metric_rows = [
    evaluate_predictions(
        y_true,
        baseline,
        "baseline_course_mean",
    )
]

model_list = [m.strip() for m in args.models.split(",") if m.strip()]

for model_name in model_list:
    print("\n" + "=" * 80)
    print(f"Running model: {model_name}")
    print("=" * 80)

    if model_name == "tfidf_ridge":
        pred = run_tfidf_ridge(train_df, val_df, args)
    elif model_name == "tfidf_mlp":
        pred = run_tfidf_mlp(train_df, val_df, args)
    elif model_name == "sbert_ridge":
        pred = run_sbert_ridge(train_df, val_df, args)
    elif model_name == "bert_finetune":
        pred = run_bert_finetune(train_df, val_df, args)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pred_col = f"pred_{model_name}"
    val_out[pred_col] = pred

    val_out, scaled_col = add_scaled_to_course_mean(
        val_out,
        pred_col=pred_col,
        baseline_col="baseline_course_mean_rating",
    )

    metric_rows.append(
        evaluate_predictions(
            y_true,
            pred,
            model_name,
        )
    )
    metric_rows.append(
        evaluate_predictions(
            y_true,
            val_out[scaled_col].values,
            f"{model_name}_scaled_to_course_mean",
        )
    )

metrics = pd.DataFrame(metric_rows)

metrics_path = output_dir / "rating_model_comparison_metrics.csv"
pred_path = output_dir / "rating_model_comparison_predictions.csv"

metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
val_out.to_csv(pred_path, index=False, encoding="utf-8-sig")

print("\n=== Rating Model Comparison Metrics ===")
print(metrics.to_string(index=False))
print("\nsaved:")
print(metrics_path)
print(pred_path)

return metrics

def get_args():parser = argparse.ArgumentParser()

parser.add_argument(
    "--raw",
    default="/content/OptiLife/data/csv/raw_everytime_reviews.csv",
)
parser.add_argument(
    "--output-dir",
    default="outputs_rating_model_comparison",
)
parser.add_argument(
    "--models",
    default="tfidf_ridge,tfidf_mlp,sbert_ridge,bert_finetune",
    help="comma-separated: tfidf_ridge,tfidf_mlp,sbert_ridge,bert_finetune",
)

parser.add_argument("--val-size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true")

# TF-IDF
parser.add_argument("--max-features", type=int, default=5000)
parser.add_argument("--min-df", type=int, default=2)
parser.add_argument("--ridge-alpha", type=float, default=1.0)

# MLP
parser.add_argument("--mlp-hidden-dim", type=int, default=64)
parser.add_argument("--mlp-alpha", type=float, default=5e-4)
parser.add_argument("--mlp-lr", type=float, default=5e-4)
parser.add_argument("--mlp-max-iter", type=int, default=800)
parser.add_argument("--mlp-patience", type=int, default=30)

# SBERT
parser.add_argument(
    "--sbert-model",
    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
parser.add_argument("--sbert-batch-size", type=int, default=64)

# BERT fine-tuning
parser.add_argument(
    "--bert-model",
    default="klue/bert-base",
)
parser.add_argument("--bert-epochs", type=int, default=1)
parser.add_argument("--bert-batch-size", type=int, default=16)
parser.add_argument("--bert-max-length", type=int, default=128)
parser.add_argument("--bert-lr", type=float, default=2e-5)
parser.add_argument("--bert-weight-decay", type=float, default=0.01)
parser.add_argument("--bert-warmup-ratio", type=float, default=0.1)
parser.add_argument("--bert-dropout", type=float, default=0.1)
parser.add_argument("--bert-grad-clip", type=float, default=1.0)

args, _ = parser.parse_known_args()
return args

if name == "main":run_experiment(get_args())
