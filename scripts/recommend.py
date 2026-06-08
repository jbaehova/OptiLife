import argparse
import itertools
import math
import os
import re

import pandas as pd

SLOT_PATTERN = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri)\s+(\d{2}):(\d{2})-(\d{2}):(\d{2})$"
)

DEFAULT_DATA = ""  # 과목 CSV input 파일 경로
DEFAULT_REVIEW_DATA = ""  # Deprecated: 추천은 courses.csv의 에브리타임 평가값을 사용
DEFAULT_OUTPUT = ""  # TXT output 경로

DEFAULT_SCORE = 3.0
DEFAULT_RATING = 3.0

REQUIRED_COURSE_COLUMNS = {
    "course_name",
    "credits",
    "core",
    "time_slot",
    "rating",
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
}

LABEL_COLUMNS = [
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]

COURSE_SCORE_COLUMNS = ["rating", *LABEL_COLUMNS]

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
    "rating_weight": 1.0,
    "workload_weight": 1.1,
    "teamwork_weight": 0.8,
    "grading_weight": 0.9,
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


def parse_score(value, default=DEFAULT_SCORE, low=1.0, high=5.0):
    parsed = parse_float(value, default)
    if math.isnan(parsed):
        parsed = default
    return round(max(low, min(high, parsed)), 2)


def parse_rating(value, default=DEFAULT_RATING):
    return parse_score(value, default=default, low=0.0, high=5.0)


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
        "rating_weight",
        "workload_weight",
        "teamwork_weight",
        "grading_weight",
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


def normalize_course_record(course):
    record = dict(course)
    record["course_name"] = clean_string(record.get("course_name", ""))
    record["professor"] = clean_string(record.get("professor", ""))
    record["credits"] = int(record["credits"])
    record["core"] = parse_bool(record["core"])
    record["time_slot"] = clean_string(record["time_slot"])
    record["rating"] = parse_rating(record.get("rating", DEFAULT_RATING))
    for column in LABEL_COLUMNS:
        record[column] = parse_score(record.get(column, DEFAULT_SCORE))
    return record


def load_courses(path, review_path=None):
    _ = review_path
    courses = read_csv(path)
    validate_columns(courses, REQUIRED_COURSE_COLUMNS, path)
    records = [
        normalize_course_record(course)
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
    if not clean_string(time_slot):
        return []

    slots = []
    for raw_slot in time_slot.split(";"):
        slot = raw_slot.strip()
        if not slot:
            continue
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
        burden = course_burden(course)
        for day, _, _ in parse_slots(course["time_slot"]):
            burden_by_day[day] = burden_by_day.get(day, 0) + burden
    return burden_by_day


def course_burden(course):
    return sum(6 - float(course[column]) for column in LABEL_COLUMNS) / len(
        LABEL_COLUMNS
    )


def format_day(day):
    return DAY_LABELS.get(day, day)


def score_schedule(schedule, preferences=None):
    preferences = normalize_preferences(preferences)
    total_rating = sum(course["rating"] for course in schedule)
    total_workload = sum(course["workload_label"] for course in schedule)
    total_teamwork = sum(course["teamwork_load_label"] for course in schedule)
    total_grading = sum(course["grading_strictness_label"] for course in schedule)
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
    score += total_rating * preferences["rating_weight"]
    score += total_workload * preferences["workload_weight"]
    score += total_teamwork * preferences["teamwork_weight"]
    score += total_grading * preferences["grading_weight"]

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
    reasons.append(
        "평점 합계 "
        f"{total_rating:.2f}, 과제 {total_workload:.2f}, "
        f"조모임 {total_teamwork:.2f}, 성적 {total_grading:.2f}"
    )

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
            f"평점 {course['rating']:.2f}, 과제 {course['workload_label']:.2f}, "
            f"조모임 {course['teamwork_load_label']:.2f}, "
            f"성적 {course['grading_strictness_label']:.2f})"
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
        "충돌 없는 과목 조합을 만든 뒤 전공필수, 금요일 수업 여부, 이른 아침 수업 여부, 평점, 과제, 조모임, 성적을 기준으로 점수를 계산합니다.",
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
        help="Deprecated. Recommendation scores use course-level Everytime values in courses.csv.",
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
