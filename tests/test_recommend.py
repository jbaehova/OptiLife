import pytest

from scripts.recommend import (
    build_recommendation_diagnostics,
    find_recommendations,
    load_courses,
    score_schedule,
)


def write_courses(path, rows):
    path.write_text(
        "\n".join(
            [
                "course_name,credits,core,rating,workload_label,teamwork_load_label,grading_strictness_label,time_slot",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_courses_parses_new_everytime_metrics(tmp_path):
    path = tmp_path / "courses.csv"
    write_courses(
        path,
        [
            "자료구조,3,true,4.25,2.4,5,3.1,Mon 09:00-10:30",
        ],
    )

    course = load_courses(path).to_dict("records")[0]

    assert course["rating"] == 4.25
    assert course["workload_label"] == 2.4
    assert course["teamwork_load_label"] == 5.0
    assert course["grading_strictness_label"] == 3.1


def test_load_courses_requires_course_level_metrics(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_text(
        "course_name,credits,core,time_slot\n자료구조,3,true,Mon 09:00-10:30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grading_strictness_label"):
        load_courses(path)


def test_recommendation_prefers_higher_everytime_scores():
    base = {
        "credits": 3,
        "core": False,
        "time_slot": "",
    }
    good = {
        **base,
        "course_name": "좋은수업",
        "rating": 5.0,
        "workload_label": 5.0,
        "teamwork_load_label": 5.0,
        "grading_strictness_label": 5.0,
    }
    bad = {
        **base,
        "course_name": "부담수업",
        "rating": 1.0,
        "workload_label": 1.0,
        "teamwork_load_label": 1.0,
        "grading_strictness_label": 1.0,
    }

    recommendations = find_recommendations(
        [bad, good],
        min_credits=3,
        max_credits=3,
        limit=1,
        preferences={
            "avoid_early": False,
            "avoid_friday_afternoon": False,
            "balance_days": False,
            "rating_weight": 1,
            "workload_weight": 1,
            "teamwork_weight": 1,
            "grading_weight": 1,
        },
    )

    assert recommendations[0][1][0]["course_name"] == "좋은수업"


def test_low_scores_increase_daily_burden_penalty():
    light = {
        "course_name": "가벼운수업",
        "credits": 3,
        "core": False,
        "rating": 3.0,
        "workload_label": 5.0,
        "teamwork_load_label": 5.0,
        "grading_strictness_label": 5.0,
        "time_slot": "Mon 09:00-10:00",
    }
    heavy = {
        **light,
        "course_name": "부담수업",
        "workload_label": 1.0,
        "teamwork_load_label": 1.0,
        "grading_strictness_label": 1.0,
    }

    preferences = {
        "avoid_early": False,
        "avoid_friday_afternoon": False,
        "balance_days": True,
        "daily_burden_limit": 1,
        "daily_burden_penalty": 10,
        "rating_weight": 0,
        "workload_weight": 0,
        "teamwork_weight": 0,
        "grading_weight": 0,
    }

    light_score, _ = score_schedule([light], preferences=preferences)
    heavy_score, _ = score_schedule([heavy], preferences=preferences)

    assert light_score > heavy_score


def make_course(
    name,
    course_id,
    time_slot,
    credits=3,
    core=False,
    category="전공심화",
    rating=3.0,
):
    return {
        "course_name": name,
        "course_id": course_id,
        "section": "41",
        "credits": credits,
        "core": core,
        "category": category,
        "rating": rating,
        "workload_label": 3.0,
        "teamwork_load_label": 3.0,
        "grading_strictness_label": 3.0,
        "time_slot": time_slot,
    }


def test_core_count_range_filters_recommendations():
    courses = [
        make_course("필수1", "CORE1", "Mon 09:00-10:00", core=True),
        make_course("필수2", "CORE2", "Tue 09:00-10:00", core=True),
        make_course("선택1", "ELEC1", "Wed 09:00-10:00"),
    ]

    recommendations = find_recommendations(
        courses,
        min_credits=6,
        max_credits=6,
        limit=3,
        core_min_count=1,
        core_max_count=1,
    )

    assert recommendations
    for _, schedule, _ in recommendations:
        assert sum(1 for course in schedule if course["core"]) == 1


def test_category_filter_limits_candidates():
    courses = [
        make_course(
            "전공코어",
            "CORE1",
            "Mon 09:00-10:00",
            core=True,
            category="전공코어",
        ),
        make_course("교양", "GED1", "Tue 09:00-10:00", category="교양"),
    ]

    recommendations = find_recommendations(
        courses,
        min_credits=3,
        max_credits=3,
        limit=3,
        categories=["교양"],
    )

    assert recommendations
    assert recommendations[0][1][0]["course_name"] == "교양"


def test_same_course_id_sections_are_not_duplicated():
    first_section = make_course("자료구조", "CSE1", "Mon 09:00-10:00", rating=5.0)
    second_section = {
        **make_course("자료구조", "CSE1", "Tue 09:00-10:00", rating=4.0),
        "section": "42",
    }
    other = make_course("알고리즘", "CSE2", "Wed 09:00-10:00")

    recommendations = find_recommendations(
        [first_section, second_section, other],
        min_credits=6,
        max_credits=6,
        limit=1,
    )

    selected_ids = [course["course_id"] for course in recommendations[0][1]]
    assert selected_ids.count("CSE1") == 1


def test_diagnostics_reports_core_min_too_high():
    courses = [
        make_course(
            "전공코어",
            "CORE1",
            "Mon 09:00-10:00",
            core=True,
            category="전공코어",
        )
    ]
    recommendations = find_recommendations(
        courses,
        min_credits=3,
        max_credits=9,
        limit=3,
        core_min_count=2,
        core_max_count=3,
        categories=["전공코어"],
    )

    diagnostics = build_recommendation_diagnostics(
        courses,
        min_credits=3,
        max_credits=9,
        recommendations=recommendations,
        core_min_count=2,
        core_max_count=3,
        categories=["전공코어"],
    )

    assert "CORE_MIN_TOO_HIGH" in {
        item["code"] for item in diagnostics["blocking"]
    }


def test_diagnostics_reports_time_conflict_bottleneck():
    courses = [
        make_course("A", "A", "Mon 09:00-10:00"),
        make_course("B", "B", "Mon 09:30-10:30"),
    ]
    recommendations = find_recommendations(
        courses,
        min_credits=6,
        max_credits=6,
        limit=3,
    )

    diagnostics = build_recommendation_diagnostics(
        courses,
        min_credits=6,
        max_credits=6,
        recommendations=recommendations,
    )

    assert diagnostics["blocking"][0]["code"] == "TIME_CONFLICT_BOTTLENECK"
