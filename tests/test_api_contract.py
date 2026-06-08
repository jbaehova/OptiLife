from app.main import courses_missing_evaluation, serialize_course


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
