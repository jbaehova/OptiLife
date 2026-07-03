# OptiLife

OptiLife is a course schedule recommendation project for university students.
It combines a small FastAPI web app, course data, review data, and machine
learning scripts that are written and trained inside this repository.

The project does not rely on an external recommendation model. It prepares
course-level signals from review data, trains models for course attributes, and
uses those signals to rank possible timetables based on user preferences.

## What It Does

- Loads course catalog data from `data/csv/courses.csv`
- Uses course attributes such as rating, workload, team project load, and
  grading strictness
- Builds valid timetable combinations without time conflicts
- Scores schedules by credits, required-course constraints, preferred free days,
  early-class avoidance, Friday-afternoon avoidance, and daily workload balance
- Serves a simple browser UI through FastAPI
- Includes scripts for weak-labeling review data and training ML models directly

## Machine Learning

OptiLife includes its own model-building workflow for lecture review data.

- `scripts/label_everytime_reviews.py` weak-labels raw review text into
  workload, teamwork, and grading-strictness labels.
- `scripts/train_difficulty_model.py` trains TF-IDF and logistic-regression
  classifiers for review labels.
- `scripts/train_course_attribute_model.py` fine-tunes a BERT-based regression
  model to predict course-level attributes from review text.
- `scripts/evaluation/` contains cross-validation experiments and saved
  comparison results.

The recommendation app reads prepared course-level values from
`data/csv/courses.csv`. Model training and evaluation are offline steps that
produce or validate those values.

## Project Structure

```text
app/
  main.py              FastAPI server and recommendation API
  static/              HTML, CSS, and JavaScript UI
data/csv/
  courses.csv          Course catalog with recommendation attributes
  raw_everytime_reviews.csv
  labeled_everytime_reviews.csv
scripts/
  recommend.py         Timetable search and scoring logic
  label_everytime_reviews.py
  train_difficulty_model.py
  train_course_attribute_model.py
  evaluation/          Model evaluation scripts and result files
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

If you use `uv`:

```bash
uv sync
```

The BERT training script may require additional ML packages that are not needed
for the web app, such as `torch`, `transformers`, and `scipy`.

## Run the Web App

```bash
python3 -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

By default, the server uses:

```text
data/csv/courses.csv
```

To use another course CSV:

```bash
OPTILIFE_COURSES_CSV=/path/to/courses.csv python3 -m uvicorn app.main:app --reload
```

## Course Data

The recommender expects these core columns in the course CSV:

```csv
course_name,credits,core,rating,workload_label,teamwork_load_label,grading_strictness_label,time_slot
Data Structures,3,true,4.2,2.4,4.7,3.1,Mon 09:00-10:30;Wed 09:00-10:30
```

`time_slot` uses this format:

```text
Mon|Tue|Wed|Thu|Fri HH:MM-HH:MM
```

Multiple class blocks are separated with semicolons.

## API

- `GET /api/health` returns app status, CSV path, row count, and data readiness.
- `GET /api/courses` returns course records and prepared evaluation values.
- `POST /api/recommend` returns recommended schedules and failure diagnostics.

## CLI Usage

Run the recommendation logic without the web app:

```bash
python3 scripts/recommend.py \
  --data data/csv/courses.csv \
  --output outputs/recommendations.txt
```

Weak-label raw review data:

```bash
python3 scripts/label_everytime_reviews.py \
  --input data/csv/raw_everytime_reviews.csv \
  --output data/csv/labeled_everytime_reviews.csv \
  --summary outputs/review_labeling_summary.txt \
  --merge-existing
```

Train TF-IDF review-label models:

```bash
python3 scripts/train_difficulty_model.py \
  --input data/csv/labeled_everytime_reviews.csv \
  --output-dir outputs/review_label_model
```

## Verification

```bash
python3 -m compileall app scripts
python3 -m pytest
```
