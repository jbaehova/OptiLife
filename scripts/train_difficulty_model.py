import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_INPUT = ""  # 라벨링된 원문 CSV input 파일 경로
DEFAULT_OUTPUT_DIR = ""  # 모델 학습 결과 파일 저장 경로

try:
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise SystemExit(
        "Missing ML dependency. Run: python3 -m pip install -r requirements.txt"
    ) from exc


OUTPUT_FILENAMES = {
    "metrics": "difficulty_model_metrics.csv",
    "history": "difficulty_training_history.csv",
    "plot": "difficulty_overfitting_curve.png",
    "lecture_scores": "lecture_difficulty_scores.csv",
    "model": "difficulty_tfidf_logreg_model.joblib",
}


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_dataset(path):
    data = pd.read_csv(path)
    required_columns = {"raw_review_text", "difficulty_label"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.fillna({"professor": ""})
    data = data.dropna(subset=["raw_review_text", "difficulty_label"])
    data["difficulty_label"] = data["difficulty_label"].astype(int)
    if "workload_label" in data.columns:
        data["workload_label"] = data["workload_label"].astype(int)
    return data


def evaluate(name, model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
    }


def train_baselines(x_train, x_test, y_train, y_test):
    rows = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    rows.append(evaluate("majority_class", dummy, x_test, y_test))

    logistic = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    logistic.fit(x_train, y_train)
    rows.append(evaluate("tfidf_logistic_regression", logistic, x_test, y_test))

    return rows, logistic


def train_mlp_with_history(x_train, x_test, y_train, y_test, epochs):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)
    x_train_vec = vectorizer.fit_transform(x_train).toarray()
    x_test_vec = vectorizer.transform(x_test).toarray()
    classes = sorted(set(y_train) | set(y_test))

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        random_state=41,
        max_iter=1,
    )

    history = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(1, epochs + 1):
            model.partial_fit(x_train_vec, y_train, classes=classes)

            train_prob = model.predict_proba(x_train_vec)
            test_prob = model.predict_proba(x_test_vec)
            train_pred = model.predict(x_train_vec)
            test_pred = model.predict(x_test_vec)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(log_loss(y_train, train_prob, labels=classes), 6),
                    "test_loss": round(log_loss(y_test, test_prob, labels=classes), 6),
                    "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
                    "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
                }
            )

    final_metrics = {
        "model": "tfidf_mlp",
        "accuracy": history[-1]["test_accuracy"],
        "macro_f1": round(
            f1_score(y_test, model.predict(x_test_vec), average="macro", zero_division=0),
            4,
        ),
    }

    return vectorizer, model, history, final_metrics


def write_dict_csv(path, rows):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_overfitting_plot(path, history):
    ensure_dir(path)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.title("TF-IDF + MLP Difficulty Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_lecture_scores(path, data, model):
    texts = data["raw_review_text"].astype(str).tolist()
    pred = model.predict(texts)

    grouped = defaultdict(list)
    for row, predicted in zip(data.to_dict("records"), pred):
        grouped[row["lecture_id"]].append((row, int(predicted)))

    rows = []
    for lecture_id, items in grouped.items():
        source_rows = [item[0] for item in items]
        predicted_labels = [item[1] for item in items]
        rows.append(
            {
                "lecture_id": lecture_id,
                "course_name": source_rows[0]["course_name"],
                "professor": source_rows[0].get("professor", ""),
                "review_count": len(items),
                "avg_predicted_difficulty": round(
                    sum(predicted_labels) / len(predicted_labels), 3
                ),
                "avg_labeled_difficulty": round(
                    sum(int(row["difficulty_label"]) for row in source_rows)
                    / len(source_rows),
                    3,
                ),
                "avg_workload_label": (
                    round(
                        sum(int(row["workload_label"]) for row in source_rows)
                        / len(source_rows),
                        3,
                    )
                    if "workload_label" in source_rows[0]
                    else ""
                ),
            }
        )

    rows.sort(key=lambda row: (row["course_name"], row["lecture_id"]))
    write_dict_csv(path, rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a TF-IDF based lecture-review difficulty classifier. "
            "The input CSV must contain raw_review_text and difficulty_label columns."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        metavar="PATH",
        help="CSV file path containing labeled reviews for model training.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help="Directory where metrics, plot, lecture scores, and model files are written.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=41)
    args = parser.parse_args()

    if not args.input:
        parser.error("--input 경로를 지정하거나 DEFAULT_INPUT에 라벨링된 리뷰 CSV 경로를 입력하세요.")
    if not args.output_dir:
        parser.error("--output-dir 경로를 지정하거나 DEFAULT_OUTPUT_DIR에 결과 폴더 경로를 입력하세요.")

    output_dir = Path(args.output_dir)
    output_paths = {
        key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()
    }

    data = read_dataset(args.input)
    x = data["raw_review_text"].astype(str)
    y = data["difficulty_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    metrics, logistic_model = train_baselines(x_train, x_test, y_train, y_test)
    vectorizer, mlp, history, mlp_metrics = train_mlp_with_history(
        x_train.tolist(),
        x_test.tolist(),
        y_train.tolist(),
        y_test.tolist(),
        args.epochs,
    )
    metrics.append(mlp_metrics)

    write_dict_csv(output_paths["metrics"], metrics)
    write_dict_csv(output_paths["history"], history)
    write_overfitting_plot(output_paths["plot"], history)
    write_lecture_scores(output_paths["lecture_scores"], data, logistic_model)
    ensure_dir(output_paths["model"])
    joblib.dump(logistic_model, output_paths["model"])

    print(f"wrote metrics to {output_paths['metrics']}")
    print(f"wrote history to {output_paths['history']}")
    print(f"wrote overfitting plot to {output_paths['plot']}")
    print(f"wrote lecture scores to {output_paths['lecture_scores']}")
    print(f"wrote model to {output_paths['model']}")


if __name__ == "__main__":
    main()
