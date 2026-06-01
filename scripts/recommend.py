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
DEFAULT_OUTPUT = ""  # TXT output 경로


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_courses(path):
    courses = pd.read_csv(path)
    courses["credits"] = courses["credits"].astype(int)
    courses["core"] = courses["core"].apply(parse_bool)
    courses["difficulty_label"] = courses["difficulty_label"].astype(int)
    courses["workload_label"] = courses["workload_label"].astype(int)
    return courses


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


def score_schedule(schedule):
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
    early_slots = [slot for slot in all_slots if slot[1] < 10 * 60]
    friday_afternoon_slots = [slot for slot in friday_slots if slot[1] >= 12 * 60]

    score = 100.0
    score += core_count * 8
    score += credits * 1.5
    score -= total_difficulty * 1.3
    score -= total_workload * 1.1
    score -= len(early_slots) * 2.0
    score -= len(friday_afternoon_slots) * 5.0

    if not friday_slots:
        score += 8
    if not early_slots:
        score += 4

    burden_penalty = 0
    for burden in daily_burden(schedule).values():
        burden_penalty += max(0, burden - 9)
    score -= burden_penalty * 1.5

    reasons = []
    reasons.append(f"전공필수 {core_count}개 포함")
    reasons.append(f"총 {credits}학점")
    if not friday_slots:
        reasons.append("금요일 수업 없음")
    elif friday_afternoon_slots:
        reasons.append("금요일 오후 수업 있음")
    if not early_slots:
        reasons.append("오전 10시 이전 수업 없음")
    if burden_penalty > 0:
        reasons.append("특정 요일 부담이 조금 몰림")
    reasons.append(f"난이도 합계 {total_difficulty}, 과제량 합계 {total_workload}")

    return score, reasons


def find_recommendations(courses, min_credits, max_credits, limit):
    course_records = to_course_records(courses)
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
            score, reasons = score_schedule(schedule)
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
    parser.add_argument("--min-credits", type=int, default=15)
    parser.add_argument("--max-credits", type=int, default=18)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if not args.data:
        parser.error("--data 경로를 지정하거나 DEFAULT_DATA에 과목 CSV 경로를 입력하세요.")
    if not args.output:
        parser.error("--output 경로를 지정하거나 DEFAULT_OUTPUT에 결과 TXT 경로를 입력하세요.")

    courses = load_courses(args.data)
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
