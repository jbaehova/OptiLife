import pytest

from scripts.recommend import find_recommendations, load_courses, score_schedule


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
