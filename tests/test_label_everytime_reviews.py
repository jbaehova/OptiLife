from scripts.label_everytime_reviews import (
    label_grading_strictness,
    label_rows,
    label_teamwork,
    label_workload,
)


def test_review_label_direction_uses_higher_as_better():
    assert label_workload("과제가 없고 부담이 없습니다") == 5
    assert label_workload("과제가 매주 많고 할 게 많습니다") == 1
    assert label_workload("평범한 강의입니다") == 3

    assert label_teamwork("조모임이 없습니다") == 5
    assert label_teamwork("팀플과 조별 발표가 많습니다") == 1

    assert label_grading_strictness("성적을 잘 주고 학점이 후합니다") == 5
    assert label_grading_strictness("성적이 깐깐하고 학점이 짭니다") == 1


def test_label_rows_uses_new_columns_and_removes_old_difficulty():
    rows = [
        {
            "review_id": "1",
            "source_url": "",
            "lecture_id": "100",
            "course_name": "자료구조",
            "professor": "김교수",
            "semester": "26년 2학기 수강자",
            "rating": "5",
            "raw_review_text": "과제가 없고 조모임도 없습니다. 학점도 잘 줍니다.",
            "difficulty_label": "1",
        }
    ]

    labeled = label_rows(rows)[0]

    assert "difficulty_label" not in labeled
    assert labeled["semester"] == "2026년 2학기"
    assert labeled["workload_label"] == 5
    assert labeled["teamwork_load_label"] == 5
    assert labeled["grading_strictness_label"] == 5
