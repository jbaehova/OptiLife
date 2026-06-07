import argparse
import itertools
import math
import os
import re
from pathlib import Path

import pandas as pd

SLOT_PATTERN = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri)\s+(\d{2}):(\d{2})-(\d{2}):(\d{2})$"
)

DEFAULT_DATA = ""  # 과목 CSV input 파일 경로
DEFAULT_REVIEW_DATA = ""  # 라벨링된 Everytime 리뷰 CSV input 파일 경로
DEFAULT_OUTPUT = ""  # TXT output 경로

DEFAULT_LABEL = 3

REQUIRED_COURSE_COLUMNS = {
    "course_name",
    "credits",
    "core",
    "time_slot",
}

LABEL_COLUMNS = [
    "difficulty_label",
    "workload_label",
    "grading_strictness_label",
]

DAY_LABELS = {
    "Mon": "월요일",
    "Tue": "화요일",
    "Wed": "수요일",
    "Thu": "목요일",
    "Fri": "금요일",
}

DAY_ALIASES = {
    "Mon": "Mon",
    "Monday": "Mon",
    "월": "Mon",
    "월요일": "Mon",
    "Tue": "Tue",
    "Tuesday": "Tue",
    "화": "Tue",
    "화요일": "Tue",
    "Wed": "Wed",
    "Wednesday": "Wed",
    "수": "Wed",
    "수요일": "Wed",
    "Thu": "Thu",
    "Thursday": "Thu",
    "목": "Thu",
    "목요일": "Thu",
    "Fri": "Fri",
    "Friday": "Fri",
    "금": "Fri",
    "금요일": "Fri",
}

DEFAULT_PREFERENCES = {
    "preferred_free_days": ["Fri"],
    "avoid_early": True,
    "avoid_friday_afternoon": True,
    "balance_days": True,
    "early_cutoff": "10:00",
    "core_weight": 8.0,
    "credit_weight": 1.5,
    "difficulty_weight": 1.3,
    "workload_weight": 1.1,
    "free_day_bonus": 8.0,
    "early_penalty": 2.0,
    "no_early_bonus": 4.0,
    "friday_afternoon_penalty": 5.0,
    "daily_burden_limit": 9.0,
    "daily_burden_penalty": 1.5,
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_day(value):
    return DAY_ALIASES.get(str(value).strip())


def clean_string(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_label(value, default=DEFAULT_LABEL):
    parsed = parse_float(value, default)
    if math.isnan(parsed):
        parsed = default
    return int(max(1, min(5, math.floor(parsed + 0.5))))


def parse_time_to_minutes(value, default):
    match = re.match(r"^(\d{1,2}):(\d{2})$", str(value).strip())
    if not match:
        return parse_time_to_minutes(default, "10:00")
    hour, minute = match.groups()
    return to_minutes(hour, minute)


def normalize_preferences(preferences=None):
    normalized = dict(DEFAULT_PREFERENCES)
    if preferences:
        for key in DEFAULT_PREFERENCES:
            if key in preferences and preferences[key] is not None:
                normalized[key] = preferences[key]

    free_days = normalized["preferred_free_days"]
    if isinstance(free_days, str):
        free_days = free_days.split(",")
    normalized["preferred_free_days"] = [
        day
        for day in (normalize_day(day) for day in free_days)
        if day in DAY_LABELS
    ]

    for key in ["avoid_early", "avoid_friday_afternoon", "balance_days"]:
        normalized[key] = parse_bool(normalized[key])

    for key in [
        "core_weight",
        "credit_weight",
        "difficulty_weight",
        "workload_weight",
        "free_day_bonus",
        "early_penalty",
        "no_early_bonus",
        "friday_afternoon_penalty",
        "daily_burden_limit",
        "daily_burden_penalty",
    ]:
        normalized[key] = parse_float(normalized[key], DEFAULT_PREFERENCES[key])

    normalized["early_cutoff_minutes"] = parse_time_to_minutes(
        normalized["early_cutoff"],
        DEFAULT_PREFERENCES["early_cutoff"],
    )
    return normalized


def validate_columns(frame, required_columns, path):
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def summarize_label_group(group):
    summary = {
        "review_count": int(len(group)),
    }
    for column in LABEL_COLUMNS:
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        average = float(values.mean()) if not values.empty else DEFAULT_LABEL
        summary[column] = parse_label(average)
        summary[f"{column}_average"] = round(average, 2)
    return summary


def load_review_aggregates(path):
    if not path:
        return {"by_course_professor": {}, "by_course": {}}

    review_path = Path(path)
    if not review_path.exists():
        raise FileNotFoundError(f"Labeled review CSV not found: {path}")

    reviews = read_csv(review_path)
    required_columns = {"course_name", *LABEL_COLUMNS}
    validate_columns(reviews, required_columns, review_path)

    reviews = reviews.copy()
    reviews["course_name"] = reviews["course_name"].apply(clean_string)
    if "professor" not in reviews.columns:
        reviews["professor"] = ""
    reviews["professor"] = reviews["professor"].apply(clean_string)
    reviews = reviews[reviews["course_name"] != ""]

    by_course_professor = {}
    for (course_name, professor), group in reviews.groupby(
        ["course_name", "professor"],
        dropna=False,
    ):
        by_course_professor[(course_name, professor)] = summarize_label_group(group)

    by_course = {}
    for course_name, group in reviews.groupby("course_name", dropna=False):
        by_course[course_name] = summarize_label_group(group)

    return {
        "by_course_professor": by_course_professor,
        "by_course": by_course,
    }


def lookup_review_summary(course, review_aggregates):
    course_name = clean_string(course.get("course_name", ""))
    professor = clean_string(course.get("professor", ""))
    return (
        review_aggregates["by_course_professor"].get((course_name, professor))
        or review_aggregates["by_course"].get(course_name)
    )


def course_csv_label_summary(course):
    if not all(
        column in course and clean_string(course[column]) != ""
        for column in LABEL_COLUMNS[:2]
    ):
        return None

    summary = {"review_count": 0}
    for column in LABEL_COLUMNS:
        if column in course and not pd.isna(course[column]):
            label = parse_label(course[column])
        else:
            label = DEFAULT_LABEL
        summary[column] = label
        summary[f"{column}_average"] = float(label)
    return summary


def normalize_course_record(course, review_aggregates):
    record = dict(course)
    record["course_name"] = clean_string(record.get("course_name", ""))
    record["professor"] = clean_string(record.get("professor", ""))
    record["credits"] = int(record["credits"])
    record["core"] = parse_bool(record["core"])
    record["time_slot"] = clean_string(record["time_slot"])

    review_summary = lookup_review_summary(record, review_aggregates)
    csv_summary = course_csv_label_summary(record)
    label_source = "labeled_reviews"
    summary = review_summary
    if summary is None:
        summary = csv_summary
        label_source = "course_csv"
    if summary is None:
        summary = {
            "review_count": 0,
            "difficulty_label": DEFAULT_LABEL,
            "difficulty_label_average": float(DEFAULT_LABEL),
            "workload_label": DEFAULT_LABEL,
            "workload_label_average": float(DEFAULT_LABEL),
            "grading_strictness_label": DEFAULT_LABEL,
            "grading_strictness_label_average": float(DEFAULT_LABEL),
        }
        label_source = "default"

    for column in LABEL_COLUMNS:
        record[column] = int(summary[column])
        record[f"{column}_average"] = float(summary[f"{column}_average"])
    record["review_count"] = int(summary["review_count"])
    record["label_source"] = label_source
    return record


def load_courses(path, review_path=None):
    courses = read_csv(path)
    validate_columns(courses, REQUIRED_COURSE_COLUMNS, path)
    review_aggregates = load_review_aggregates(review_path)
    records = [
        normalize_course_record(course, review_aggregates)
        for course in courses.to_dict("records")
    ]
    return pd.DataFrame(records)


def to_course_records(courses):
    if isinstance(courses, pd.DataFrame):
        return courses.to_dict("records")
    return list(courses)


def to_minutes(hour, minute):
    return int(hour) * 60 + int(minute)


def parse_slots(time_slot):
    slots = []
    for raw_slot in time_slot.split(";"):
        slot = raw_slot.strip()
        match = SLOT_PATTERN.match(slot)
        if not match:
            raise ValueError(f"Invalid time slot: {slot}")
        day, start_h, start_m, end_h, end_m = match.groups()
        slots.append((day, to_minutes(start_h, start_m), to_minutes(end_h, end_m)))
    return slots


def overlaps(left, right):
    left_day, left_start, left_end = left
    right_day, right_start, right_end = right
    return left_day == right_day and left_start < right_end and right_start < left_end


def has_conflict(schedule):
    slots = []
    for course in schedule:
        for slot in parse_slots(course["time_slot"]):
            for existing_slot, _ in slots:
                if overlaps(slot, existing_slot):
                    return True
            slots.append((slot, course["course_name"]))
    return False


def daily_burden(schedule):
    burden_by_day = {}
    for course in schedule:
        burden = (course["difficulty_label"] + course["workload_label"]) / 2
        for day, _, _ in parse_slots(course["time_slot"]):
            burden_by_day[day] = burden_by_day.get(day, 0) + burden
    return burden_by_day


def format_day(day):
    return DAY_LABELS.get(day, day)


def score_schedule(schedule, preferences=None):
    preferences = normalize_preferences(preferences)
    total_difficulty = sum(course["difficulty_label"] for course in schedule)
    total_workload = sum(course["workload_label"] for course in schedule)
    core_count = sum(1 for course in schedule if course["core"])
    credits = sum(course["credits"] for course in schedule)

    all_slots = [
        slot
        for course in schedule
        for slot in parse_slots(course["time_slot"])
    ]
    friday_slots = [slot for slot in all_slots if slot[0] == "Fri"]
    early_slots = [
        slot for slot in all_slots if slot[1] < preferences["early_cutoff_minutes"]
    ]
    friday_afternoon_slots = [slot for slot in friday_slots if slot[1] >= 12 * 60]

    score = 100.0
    score += core_count * preferences["core_weight"]
    score += credits * preferences["credit_weight"]
    score -= total_difficulty * preferences["difficulty_weight"]
    score -= total_workload * preferences["workload_weight"]

    if preferences["avoid_early"]:
        score -= len(early_slots) * preferences["early_penalty"]
        if not early_slots:
            score += preferences["no_early_bonus"]

    if preferences["avoid_friday_afternoon"]:
        score -= (
            len(friday_afternoon_slots)
            * preferences["friday_afternoon_penalty"]
        )

    free_day_reasons = []
    for day in preferences["preferred_free_days"]:
        day_slots = [slot for slot in all_slots if slot[0] == day]
        if not day_slots:
            score += preferences["free_day_bonus"]
            free_day_reasons.append(f"{format_day(day)} 공강")

    burden_penalty = 0
    if preferences["balance_days"]:
        for burden in daily_burden(schedule).values():
            burden_penalty += max(0, burden - preferences["daily_burden_limit"])
        score -= burden_penalty * preferences["daily_burden_penalty"]

    reasons = []
    reasons.append(f"전공필수 {core_count}개 포함")
    reasons.append(f"총 {credits}학점")
    reasons.extend(free_day_reasons)
    if preferences["avoid_friday_afternoon"] and friday_afternoon_slots:
        reasons.append("금요일 오후 수업 있음")
    if preferences["avoid_early"] and not early_slots:
        reasons.append(
            f"{format_minutes(preferences['early_cutoff_minutes'])} 이전 수업 없음"
        )
    if burden_penalty > 0:
        reasons.append("특정 요일 부담이 조금 몰림")
    reasons.append(f"난이도 합계 {total_difficulty}, 과제량 합계 {total_workload}")

    return score, reasons


def find_recommendations(courses, min_credits, max_credits, limit, preferences=None):
    course_records = to_course_records(courses)
    if not course_records:
        return []

    min_course_credits = min(course["credits"] for course in course_records)
    max_course_credits = max(course["credits"] for course in course_records)
    min_count = max(1, math.ceil(min_credits / max_course_credits))
    max_count = min(len(course_records), math.floor(max_credits / min_course_credits))

    candidates = []
    for count in range(min_count, max_count + 1):
        for schedule in itertools.combinations(course_records, count):
            credits = sum(course["credits"] for course in schedule)
            if credits < min_credits or credits > max_credits:
                continue
            if has_conflict(schedule):
                continue
            score, reasons = score_schedule(schedule, preferences=preferences)
            candidates.append((score, schedule, reasons))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:limit]


def format_minutes(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_schedule(rank, score, schedule, reasons):
    lines = [f"[추천 {rank}] score={score:.2f}"]
    for course in schedule:
        core_mark = "전공필수" if course["core"] else "선택"
        lines.append(
            f"- {course['course_name']} ({course['credits']}학점, {core_mark}, "
            f"난이도 {course['difficulty_label']}, 과제량 {course['workload_label']})"
        )
        for day, start, end in parse_slots(course["time_slot"]):
            lines.append(f"  - {day} {format_minutes(start)}-{format_minutes(end)}")
    lines.append("추천 이유: " + ", ".join(reasons))
    return "\n".join(lines)


def build_report(recommendations):
    lines = [
        "시간표 추천 프로토타입 결과",
        "",
        "현재 버전은 PyTorch 학습 모델이 아니라, 데이터 구조와 추천 점수식을 먼저 검증하기 위한 간단한 프로토타입입니다.",
        "충돌 없는 과목 조합을 만든 뒤 전공필수, 금요일 수업 여부, 이른 아침 수업 여부, 난이도, 과제량을 기준으로 점수를 계산합니다.",
        "",
    ]
    for index, (score, schedule, reasons) in enumerate(recommendations, start=1):
        lines.append(format_schedule(index, score, schedule, reasons))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        metavar="PATH",
        help="CSV file path containing candidate courses for schedule recommendation.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Text file path to write the recommendation report.",
    )
    parser.add_argument(
        "--reviews",
        default=DEFAULT_REVIEW_DATA,
        metavar="PATH",
        help="CSV file path containing labeled Everytime reviews.",
    )
    parser.add_argument("--min-credits", type=int, default=15)
    parser.add_argument("--max-credits", type=int, default=18)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if not args.data:
        parser.error("--data 경로를 지정하거나 DEFAULT_DATA에 과목 CSV 경로를 입력하세요.")
    if not args.output:
        parser.error("--output 경로를 지정하거나 DEFAULT_OUTPUT에 결과 TXT 경로를 입력하세요.")

    courses = load_courses(args.data, args.reviews or None)
    recommendations = find_recommendations(
        courses,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        limit=args.limit,
    )
    report = build_report(recommendations)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(report)
    print(report)


if __name__ == "__main__":
    main()
