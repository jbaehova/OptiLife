# Data Contract

OptiLife app/API uses `courses.csv` under `data/csv/`. The review CSV contracts below are for offline scripts that prepare course-level values before they are copied into `courses.csv`.

## `courses.csv`

Required recommendation columns:

```csv
course_name,credits,core,rating,workload_label,teamwork_load_label,grading_strictness_label,time_slot
```

Additional course metadata such as `course_id`, `academic_year`, `semester`, `department`, `section`, `professor`, `category`, `campus`, `classroom`, `capacity`, and `source` may be present.

- `rating`: Everytime star rating, numeric `0` to `5`.
- `workload_label`: Everytime assignment distribution converted to a `1` to `5` weighted average.
- `teamwork_load_label`: Everytime team-project distribution converted to a `1` to `5` weighted average.
- `grading_strictness_label`: Everytime grading distribution converted to a `1` to `5` weighted average.
- `time_slot`: semicolon-separated `Mon|Tue|Wed|Thu|Fri HH:MM-HH:MM`.

For the three distribution-based fields, higher is better:

- 과제/조모임: `없음=5`, `보통=3`, `많음=1`.
- 성적: `너그러움=5`, `보통=3`, `깐깐함=1`.

Example: 성적이 `너그러움 57%`, `보통 43%`, `깐깐함 0%`이면 `5*0.57 + 3*0.43 + 1*0 = 4.14`.

## `raw_everytime_reviews.csv`

Raw review rows should stay as close to the collected Everytime data as possible:

```csv
review_id,source_url,lecture_id,course_name,professor,semester,rating,raw_review_text
```

`semester` stores the source review semester text. Do not add predicted labels to this file.

## `labeled_everytime_reviews.csv`

Labeled review rows add only the review-level prediction targets:

```csv
review_id,source_url,lecture_id,course_name,professor,semester,rating,raw_review_text,workload_label,teamwork_load_label,grading_strictness_label
```

The three label values are `1`, `3`, or `5` with the same direction as `courses.csv`: higher means less burden or more generous grading.
