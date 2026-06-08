import pytest

from fastapi import HTTPException

from app.main import (
    ConditionData,
    courses_missing_evaluation,
    health,
    recommend,
    serialize_course,
)


def test_serialize_course_exposes_new_metrics():
    course = serialize_course(
        {
            "course_name": "자료구조",
            "credits": 3,
            "core": True,
            "rating": 4.2,
            "workload_label": 2.4,
            "teamwork_load_label": 5.0,
            "grading_strictness_label": 3.1,
            "time_slot": "Mon 09:00-10:30",
        }
    )

    assert course["rating"] == 4.2
    assert course["workload_label"] == 2.4
    assert course["teamwork_load_label"] == 5.0
    assert course["grading_strictness_label"] == 3.1
    assert "difficulty_label" not in course


def test_courses_missing_evaluation_reports_incomplete_rows(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_text(
        "\n".join(
            [
                "course_name,section,rating,workload_label,teamwork_load_label,grading_strictness_label",
                "자료구조,41,4.2,2.4,5.0,3.1",
                "운영체제,42,,2.0,4.0,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert courses_missing_evaluation(path) == ["운영체제(42)"]


def test_health_exposes_courses_only_dataset_contract():
    payload = health()

    assert "review_data_source" not in payload
    assert "raw_reviews" not in payload["dataset"]
    assert "labeled_reviews" not in payload["dataset"]
    assert "categories" in payload["dataset"]


def test_recommend_accepts_core_range_and_returns_diagnostics():
    payload = recommend(
        ConditionData(
            min_credits=3,
            max_credits=3,
            limit=1,
            core_min_count=1,
            core_max_count=1,
            categories=["전공코어"],
        )
    )

    assert "review_data_source" not in payload
    assert "diagnostics" in payload
    assert payload["condition_data"]["core_min_count"] == 1
    assert payload["condition_data"]["core_max_count"] == 1
    assert payload["recommendations"]
    assert payload["recommendations"][0]["core_count"] == 1


def test_recommend_rejects_invalid_core_range():
    with pytest.raises(HTTPException) as exc_info:
        recommend(
            ConditionData(
                min_credits=3,
                max_credits=6,
                core_min_count=3,
                core_max_count=1,
            )
        )

    assert exc_info.value.status_code == 400
