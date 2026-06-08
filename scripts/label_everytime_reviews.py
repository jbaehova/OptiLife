import argparse
import csv
import os
import re
from collections import Counter


DEFAULT_INPUT = ""  # 원본 Everytime 리뷰 CSV 파일 경로
DEFAULT_OUTPUT = ""  # 라벨링 결과 CSV output 경로
DEFAULT_SUMMARY = ""  # 라벨링 요약 TXT output 경로

REVIEW_FIELDNAMES = [
    "review_id",
    "source_url",
    "lecture_id",
    "course_name",
    "professor",
    "semester",
    "rating",
    "raw_review_text",
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]


WORKLOAD_HEAVY_PATTERNS = [
    (r"과제.*많|숙제.*많|과제.*매주|매주.*과제", 2),
    (r"발표|ppt|피피티|프로젝트", 1),
    (r"퀴즈|숙제|homework|hw", 1),
    (r"바쁘|할 게 많|할게 많|공부량.*많|부담", 1),
]

WORKLOAD_LIGHT_PATTERNS = [
    (r"과제.*없|과제.*적|할것도 별로|할 것도 별로", 2),
    (r"널널|쾌적|부담.*없|편하게", 2),
    (r"시험.*한 번|시험.*한번", 1),
]

TEAMWORK_HEAVY_PATTERNS = [
    (r"팀플|조별|조모임|group activity|그룹", 2),
    (r"발표.*조|조.*발표|팀.*프로젝트|조별.*과제", 1),
]

TEAMWORK_LIGHT_PATTERNS = [
    (r"(팀플|조별|조모임).*없|없.*(팀플|조별|조모임)", 3),
    (r"개인.*과제|혼자.*하면|혼자.*하는", 1),
]

GRADING_STRICT_PATTERNS = [
    (r"학점.*안|성적.*안|짜게|깐깐|빡세게", 2),
    (r"상대평가|커브", 1),
]

GRADING_GENEROUS_PATTERNS = [
    (r"학점.*잘|성적.*잘|점수.*잘|후하|너그러", 2),
    (r"A\+|에이플|절대평가|보너스", 1),
]


def pattern_score(text, patterns):
    score = 0
    for pattern, weight in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
    return score


def label_from_scores(good_score, bad_score):
    if good_score > bad_score:
        return 5
    if bad_score > good_score:
        return 1
    return 3


def normalize_semester(value):
    cleaned = re.sub(r"\s+", " ", str(value)).strip().replace(" 수강자", "")
    match = re.search(r"(\d{2,4})\s*년\s*([12])\s*학기", cleaned)
    if not match:
        return cleaned

    year, term = match.groups()
    if len(year) == 2:
        year = f"20{year}"
    return f"{year}년 {term}학기"


def label_workload(text):
    return label_from_scores(
        pattern_score(text, WORKLOAD_LIGHT_PATTERNS),
        pattern_score(text, WORKLOAD_HEAVY_PATTERNS),
    )


def label_teamwork(text):
    return label_from_scores(
        pattern_score(text, TEAMWORK_LIGHT_PATTERNS),
        pattern_score(text, TEAMWORK_HEAVY_PATTERNS),
    )


def label_grading_strictness(text):
    return label_from_scores(
        pattern_score(text, GRADING_GENEROUS_PATTERNS),
        pattern_score(text, GRADING_STRICT_PATTERNS),
    )


def sanitize_row(row):
    cleaned = dict(row)
    cleaned.pop("difficulty_label", None)
    cleaned["semester"] = normalize_semester(cleaned.get("semester", ""))
    return cleaned


def label_rows(rows):
    labeled = []
    for row in rows:
        text = row["raw_review_text"]
        row = sanitize_row(row)
        row["workload_label"] = label_workload(text)
        row["teamwork_load_label"] = label_teamwork(text)
        row["grading_strictness_label"] = label_grading_strictness(text)
        labeled.append(row)
    return labeled


def read_csv(path):
    with open(path, encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_fieldnames(path):
    with open(path, encoding="utf-8-sig") as file:
        return csv.DictReader(file).fieldnames or []


def row_key(row):
    review_id = row.get("review_id", "").strip()
    if review_id:
        return ("review_id", review_id)
    return (
        "content",
        row.get("lecture_id", ""),
        row.get("semester", ""),
        row.get("rating", ""),
        row.get("raw_review_text", ""),
    )


def same_row(left, right):
    fieldnames = collect_fieldnames([left, right])
    return all(
        str(left.get(field, "")) == str(right.get(field, ""))
        for field in fieldnames
    )


def merge_rows(existing_rows, new_rows):
    merged = list(existing_rows)
    additions = []
    update_count = 0
    index_by_key = {
        row_key(row): index
        for index, row in enumerate(merged)
    }
    for row in new_rows:
        key = row_key(row)
        if key in index_by_key:
            index = index_by_key[key]
            if not same_row(merged[index], row):
                merged[index] = row
                update_count += 1
        else:
            index_by_key[key] = len(merged)
            merged.append(row)
            additions.append(row)
    return merged, additions, update_count


def collect_fieldnames(rows):
    fieldnames = list(REVIEW_FIELDNAMES)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=collect_fieldnames(rows),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, rows, fieldnames):
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writerows(rows)


def build_summary(rows):
    lines = [
        "Everytime review weak-label summary",
        "",
        f"rows: {len(rows)}",
        f"course counts: {dict(Counter(row['course_name'] for row in rows))}",
        f"workload counts: {dict(Counter(row['workload_label'] for row in rows))}",
        f"teamwork counts: {dict(Counter(row['teamwork_load_label'] for row in rows))}",
        "grading counts: "
        f"{dict(Counter(row['grading_strictness_label'] for row in rows))}",
        "labeling method: keyword weak labeling, not manual ground truth",
    ]
    return "\n".join(lines) + "\n"


def write_text(path, content):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        metavar="PATH",
        help="CSV file path containing raw Everytime reviews to label.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="CSV file path to write labeled reviews.",
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
        metavar="PATH",
        help="Text file path to write label counts and method notes.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge labeled input rows into an existing output CSV instead of replacing it.",
    )
    args = parser.parse_args()

    if not args.input:
        parser.error("--input 경로를 지정하거나 DEFAULT_INPUT에 원본 리뷰 CSV 경로를 입력하세요.")
    if not args.output:
        parser.error("--output 경로를 지정하거나 DEFAULT_OUTPUT에 저장할 CSV 경로를 입력하세요.")
    if not args.summary:
        parser.error("--summary 경로를 지정하거나 DEFAULT_SUMMARY에 요약 TXT 경로를 입력하세요.")

    rows = label_rows(read_csv(args.input))
    if args.merge_existing and os.path.exists(args.output):
        existing_rows = [sanitize_row(row) for row in read_csv(args.output)]
        existing_fieldnames = read_fieldnames(args.output)
        rows, additions, update_count = merge_rows(existing_rows, rows)
        target_fieldnames = collect_fieldnames(rows)
        if update_count == 0 and additions and target_fieldnames == existing_fieldnames:
            append_csv(args.output, additions, existing_fieldnames)
        elif update_count or additions or target_fieldnames != existing_fieldnames:
            write_csv(args.output, rows)
    else:
        write_csv(args.output, rows)

    write_text(args.summary, build_summary(rows))
    print(f"wrote {len(rows)} labeled reviews to {args.output}")


if __name__ == "__main__":
    main()
