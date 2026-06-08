import argparse
import math
import os
import re
from collections import defaultdict

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

DEFAULT_BEAM_PER_BUCKET = 25


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


def normalize_categories(categories=None):
    if not categories:
        return []
    if isinstance(categories, str):
        categories = categories.split(",")
    return [clean_string(category) for category in categories if clean_string(category)]


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


def filter_courses(courses, categories=None):
    records = to_course_records(courses)
    normalized_categories = set(normalize_categories(categories))
    if not normalized_categories:
        return records
    return [
        course
        for course in records
        if clean_string(course.get("category", "")) in normalized_categories
    ]


def course_group_key(course):
    course_id = clean_string(course.get("course_id", ""))
    if course_id:
        return course_id
    return "|".join(
        [
            clean_string(course.get("department", "")),
            clean_string(course.get("course_name", "")),
            clean_string(course.get("category", "")),
        ]
    )


def group_course_sections(courses):
    grouped = defaultdict(list)
    for course in courses:
        grouped[course_group_key(course)].append(course)

    groups = list(grouped.values())
    for group in groups:
        group.sort(key=course_section_rank, reverse=True)

    groups.sort(
        key=lambda group: (
            not any(course["core"] for course in group),
            -max(course["credits"] for course in group),
            clean_string(group[0].get("course_name", "")),
            clean_string(group[0].get("section", "")),
        )
    )
    return groups


def course_section_rank(course):
    return (
        float(course.get("rating", DEFAULT_RATING)),
        float(course.get("workload_label", DEFAULT_SCORE)),
        float(course.get("teamwork_load_label", DEFAULT_SCORE)),
        float(course.get("grading_strictness_label", DEFAULT_SCORE)),
        -len(parse_slots(course.get("time_slot", ""))),
    )


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


def slots_conflict(candidate_slots, occupied_slots):
    for slot in candidate_slots:
        for occupied_slot in occupied_slots:
            if overlaps(slot, occupied_slot):
                return True
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


def diagnostic_item(code, label, expected, actual, help_text):
    return {
        "code": code,
        "label": label,
        "expected": expected,
        "actual": actual,
        "help": help_text,
    }


def preference_warnings(schedule, preferences=None):
    preferences = normalize_preferences(preferences)
    all_slots = [
        slot
        for course in schedule
        for slot in parse_slots(course["time_slot"])
    ]
    warnings = []

    if preferences["avoid_early"]:
        early_slots = [
            slot for slot in all_slots if slot[1] < preferences["early_cutoff_minutes"]
        ]
        if early_slots:
            warnings.append(
                diagnostic_item(
                    "AVOID_EARLY_UNMET",
                    "이른 수업 회피",
                    f"{format_minutes(preferences['early_cutoff_minutes'])} 이전 수업 없음",
                    f"{len(early_slots)}개 수업이 기준보다 빠름",
                    "이른 수업 회피를 끄거나 기준 시간을 앞당기면 더 많은 시간표를 비교할 수 있습니다.",
                )
            )

    if preferences["avoid_friday_afternoon"]:
        friday_afternoon_slots = [
            slot for slot in all_slots if slot[0] == "Fri" and slot[1] >= 12 * 60
        ]
        if friday_afternoon_slots:
            warnings.append(
                diagnostic_item(
                    "FRIDAY_AFTERNOON_UNMET",
                    "금요일 오후 회피",
                    "금요일 12:00 이후 수업 없음",
                    f"{len(friday_afternoon_slots)}개 금요일 오후 수업 포함",
                    "금요일 오후 회피를 끄거나 금요일 공강 선호를 함께 낮추면 후보가 늘어납니다.",
                )
            )

    for day in preferences["preferred_free_days"]:
        day_slots = [slot for slot in all_slots if slot[0] == day]
        if day_slots:
            warnings.append(
                diagnostic_item(
                    "PREFERRED_FREE_DAY_UNMET",
                    f"{format_day(day)} 공강 선호",
                    f"{format_day(day)} 수업 없음",
                    f"{len(day_slots)}개 수업 포함",
                    "해당 요일 공강 선호를 해제하거나 다른 공강 요일을 선택하면 더 현실적인 후보가 나옵니다.",
                )
            )

    if preferences["balance_days"]:
        heavy_days = [
            (day, burden)
            for day, burden in daily_burden(schedule).items()
            if burden > preferences["daily_burden_limit"]
        ]
        if heavy_days:
            day_text = ", ".join(
                f"{format_day(day)} {burden:.1f}" for day, burden in heavy_days
            )
            warnings.append(
                diagnostic_item(
                    "DAILY_BALANCE_UNMET",
                    "요일 부담 분산",
                    f"요일별 부담 {preferences['daily_burden_limit']:.1f} 이하",
                    day_text,
                    "요일 부담 분산 선호를 끄거나 학점 범위를 낮추면 특정 요일 집중을 줄일 수 있습니다.",
                )
            )

    return warnings


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


def make_search_state(schedule, occupied_slots, credits, core_count, preferences):
    rank_score = 0 if not schedule else score_schedule(schedule, preferences)[0]
    return (rank_score, schedule, occupied_slots, credits, core_count)


def prune_search_states(states, beam_per_bucket):
    buckets = defaultdict(list)
    for state in states:
        _, _, _, credits, core_count = state
        buckets[(credits, core_count)].append(state)

    pruned = []
    for bucket in buckets.values():
        bucket.sort(key=lambda state: state[0], reverse=True)
        pruned.extend(bucket[:beam_per_bucket])
    return pruned


def find_recommendations(
    courses,
    min_credits,
    max_credits,
    limit,
    preferences=None,
    core_min_count=0,
    core_max_count=None,
    categories=None,
    beam_per_bucket=DEFAULT_BEAM_PER_BUCKET,
):
    course_records = filter_courses(courses, categories=categories)
    if not course_records:
        return []

    groups = group_course_sections(course_records)
    if core_max_count is None:
        core_max_count = len(groups)

    preferences = normalize_preferences(preferences)
    states = [make_search_state(tuple(), tuple(), 0, 0, preferences)]

    for group in groups:
        next_states = list(states)
        for _, schedule, occupied_slots, credits, core_count in states:
            for course in group:
                next_credits = credits + course["credits"]
                next_core_count = core_count + (1 if course["core"] else 0)
                if next_credits > max_credits or next_core_count > core_max_count:
                    continue

                course_slots = tuple(parse_slots(course["time_slot"]))
                if slots_conflict(course_slots, occupied_slots):
                    continue

                next_schedule = schedule + (course,)
                next_occupied_slots = occupied_slots + course_slots
                next_states.append(
                    make_search_state(
                        next_schedule,
                        next_occupied_slots,
                        next_credits,
                        next_core_count,
                        preferences,
                    )
                )

        states = prune_search_states(next_states, beam_per_bucket)

    candidates = []
    seen = set()
    for _, schedule, _, credits, core_count in states:
        if credits < min_credits or credits > max_credits:
            continue
        if core_count < core_min_count or core_count > core_max_count:
            continue

        signature = tuple(
            sorted(
                (
                    clean_string(course.get("course_id", "")),
                    clean_string(course.get("section", "")),
                    clean_string(course.get("course_name", "")),
                )
                for course in schedule
            )
        )
        if signature in seen:
            continue
        seen.add(signature)

        score, reasons = score_schedule(schedule, preferences=preferences)
        candidates.append((score, schedule, reasons))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:limit]


def max_credits_with_core_limit(groups, core_max_count):
    non_core_credits = []
    core_credits = []
    for group in groups:
        credits = max(course["credits"] for course in group)
        if any(course["core"] for course in group):
            core_credits.append(credits)
        else:
            non_core_credits.append(credits)
    return sum(non_core_credits) + sum(sorted(core_credits, reverse=True)[:core_max_count])


def min_credits_for_core_count(groups, core_min_count):
    if core_min_count <= 0:
        return 0
    core_credits = sorted(
        min(course["credits"] for course in group)
        for group in groups
        if any(course["core"] for course in group)
    )
    if len(core_credits) < core_min_count:
        return math.inf
    return sum(core_credits[:core_min_count])


def build_recommendation_diagnostics(
    courses,
    min_credits,
    max_credits,
    recommendations,
    core_min_count=0,
    core_max_count=None,
    categories=None,
):
    course_records = filter_courses(courses, categories=categories)
    selected_categories = normalize_categories(categories)
    blocking = []

    if not course_records:
        blocking.append(
            diagnostic_item(
                "NO_CATEGORY_MATCH",
                "카테고리 필터",
                ", ".join(selected_categories) if selected_categories else "전체",
                "후보 과목 0개",
                "카테고리 선택을 넓히거나 courses.csv에 해당 카테고리 과목이 있는지 확인하세요.",
            )
        )
        return {
            "status": "no_schedule",
            "blocking": blocking,
            "warnings": [],
        }

    groups = group_course_sections(course_records)
    if core_max_count is None:
        core_max_count = len(groups)

    min_course_credits = min(course["credits"] for course in course_records)
    max_reachable_credits = max_credits_with_core_limit(groups, core_max_count)
    max_core_possible = sum(1 for group in groups if any(course["core"] for course in group))
    core_min_credit_floor = min_credits_for_core_count(groups, core_min_count)

    if max_credits < min_course_credits:
        blocking.append(
            diagnostic_item(
                "MAX_CREDITS_TOO_LOW",
                "최대 학점",
                f"최소 {min_course_credits}학점 이상",
                f"현재 최대 {max_credits}학점",
                "최대 학점을 가장 작은 과목 학점 이상으로 올려야 시간표를 만들 수 있습니다.",
            )
        )

    if min_credits > max_reachable_credits:
        blocking.append(
            diagnostic_item(
                "MIN_CREDITS_TOO_HIGH",
                "최소 학점",
                f"최대 {max_reachable_credits}학점 이하",
                f"현재 최소 {min_credits}학점",
                "최소 학점을 낮추거나 카테고리 필터와 전공필수 최대 개수를 완화하세요.",
            )
        )

    if core_min_count > max_core_possible:
        blocking.append(
            diagnostic_item(
                "CORE_MIN_TOO_HIGH",
                "전공필수 최소 개수",
                f"최대 {max_core_possible}개 이하",
                f"현재 최소 {core_min_count}개",
                "전공필수 최소 개수를 낮추거나 전공코어 카테고리를 포함하세요.",
            )
        )

    if core_min_credit_floor > max_credits:
        blocking.append(
            diagnostic_item(
                "MAX_CREDITS_TOO_LOW",
                "최대 학점",
                f"전공필수 {core_min_count}개를 담으려면 최소 {core_min_credit_floor}학점",
                f"현재 최대 {max_credits}학점",
                "최대 학점을 올리거나 전공필수 최소 개수를 낮추세요.",
            )
        )

    if max_reachable_credits < min_credits and core_max_count < max_core_possible:
        blocking.append(
            diagnostic_item(
                "CORE_MAX_TOO_LOW",
                "전공필수 최대 개수",
                f"최소 {min_credits}학점 도달 가능",
                f"전공필수 최대 {core_max_count}개 기준 {max_reachable_credits}학점",
                "전공필수 최대 개수를 올리거나 최소 학점을 낮추세요.",
            )
        )

    if not recommendations and not blocking:
        blocking.append(
            diagnostic_item(
                "TIME_CONFLICT_BOTTLENECK",
                "시간 충돌",
                "학점과 전공필수 조건을 동시에 만족하는 비충돌 조합",
                "후보 없음",
                "학점 범위를 낮추거나 카테고리 필터를 넓히고, 특정 과목군의 분반 시간이 겹치는지 확인하세요.",
            )
        )

    return {
        "status": "ok" if recommendations else "no_schedule",
        "blocking": [] if recommendations else blocking,
        "warnings": [],
    }


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
