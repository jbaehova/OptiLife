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
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise SystemExit(
        "Missing ML dependency. Run: python3 -m pip install -r requirements.txt"
    ) from exc


TARGET_LABELS = [
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]

OUTPUT_FILENAMES = {
    "metrics": "review_label_model_metrics.csv",
    "lecture_scores": "lecture_review_label_scores.csv",
    "models": "review_label_tfidf_logreg_models.joblib",
}


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_dataset(path):
    data = pd.read_csv(path)
    required_columns = {"raw_review_text", *TARGET_LABELS}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.fillna({"professor": "", "lecture_id": ""})
    data = data.dropna(subset=["raw_review_text", *TARGET_LABELS])
    for target in TARGET_LABELS:
        data[target] = data[target].astype(int)
    return data


def can_stratify(labels):
    return labels.value_counts().min() >= 2


def split_dataset(x, y, test_size, random_state):
    stratify = y if can_stratify(y) else None
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def evaluate(target, name, model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "target": target,
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
    }


def train_target_model(target, data, test_size, random_state):
    x = data["raw_review_text"].astype(str)
    y = data[target]
    x_train, x_test, y_train, y_test = split_dataset(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    rows = []
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    rows.append(evaluate(target, "majority_class", dummy, x_test, y_test))

    if y_train.nunique() < 2:
        rows.append(evaluate(target, "tfidf_logistic_regression", dummy, x_test, y_test))
        return rows, dummy

    logistic = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
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
    rows.append(evaluate(target, "tfidf_logistic_regression", logistic, x_test, y_test))
    return rows, logistic


def write_dict_csv(path, rows):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_lecture_scores(path, data, models):
    grouped = defaultdict(list)
    predictions = {
        target: models[target].predict(data["raw_review_text"].astype(str).tolist())
        for target in TARGET_LABELS
    }

    for index, row in enumerate(data.to_dict("records")):
        lecture_id = row.get("lecture_id", "")
        grouped[lecture_id].append((row, index))

    rows = []
    for lecture_id, items in grouped.items():
        source_rows = [item[0] for item in items]
        indexes = [item[1] for item in items]
        result = {
            "lecture_id": lecture_id,
            "course_name": source_rows[0]["course_name"],
            "professor": source_rows[0].get("professor", ""),
            "review_count": len(items),
        }
        for target in TARGET_LABELS:
            result[f"avg_predicted_{target}"] = round(
                sum(int(predictions[target][index]) for index in indexes) / len(indexes),
                3,
            )
            result[f"avg_labeled_{target}"] = round(
                sum(int(row[target]) for row in source_rows) / len(source_rows),
                3,
            )
        rows.append(result)

    rows.sort(key=lambda row: (row["course_name"], row["lecture_id"]))
    write_dict_csv(path, rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train TF-IDF based lecture-review classifiers for workload, "
            "teamwork, and grading labels."
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
        help="Directory where metrics, lecture scores, and model files are written.",
    )
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
    metrics = []
    models = {}
    for target in TARGET_LABELS:
        target_metrics, model = train_target_model(
            target,
            data,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        metrics.extend(target_metrics)
        models[target] = model

    write_dict_csv(output_paths["metrics"], metrics)
    write_lecture_scores(output_paths["lecture_scores"], data, models)
    ensure_dir(output_paths["models"])
    joblib.dump(models, output_paths["models"])

    print(f"wrote metrics to {output_paths['metrics']}")
    print(f"wrote lecture scores to {output_paths['lecture_scores']}")
    print(f"wrote models to {output_paths['models']}")


if __name__ == "__main__":
    main()
