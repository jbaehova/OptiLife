import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.recommend import (
    DAY_LABELS,
    daily_burden,
    find_recommendations,
    format_minutes,
    load_courses,
    normalize_day,
    parse_slots,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
CANONICAL_COURSES_PATH = PROJECT_ROOT / "data" / "csv" / "courses.csv"
CANONICAL_RAW_REVIEWS_PATH = PROJECT_ROOT / "data" / "csv" / "raw_everytime_reviews.csv"
CANONICAL_LABELED_REVIEWS_PATH = (
    PROJECT_ROOT / "data" / "csv" / "labeled_everytime_reviews.csv"
)
LABELING_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "review_labeling_summary.txt"
SAMPLE_COURSES_PATH = (
    PROJECT_ROOT / "scripts" / "examples" / "recommend" / "input" / "courses.csv"
)
LABELING_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "label_everytime_reviews.py"

app = FastAPI(title="OptiLife", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

UI_ASSET_PATHS = [
    STATIC_DIR / "index.html",
    STATIC_DIR / "styles.css",
    STATIC_DIR / "app.js",
]


class ConditionData(BaseModel):
    min_credits: int = Field(15, ge=1, le=30)
    max_credits: int = Field(18, ge=1, le=30)
    limit: int = Field(3, ge=1, le=10)
    preferred_free_days: List[str] = Field(default_factory=lambda: ["Fri"])
    avoid_early: bool = True
    avoid_friday_afternoon: bool = True
    balance_days: bool = True
    early_cutoff: str = "10:00"
    core_weight: float = Field(8.0, ge=0, le=20)
    difficulty_weight: float = Field(1.3, ge=0, le=5)
    workload_weight: float = Field(1.1, ge=0, le=5)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_course_path() -> Path:
    configured = os.environ.get("OPTILIFE_COURSES_CSV")
    if configured:
        return Path(configured).expanduser()

    if CANONICAL_COURSES_PATH.exists():
        return CANONICAL_COURSES_PATH

    return SAMPLE_COURSES_PATH


def resolve_raw_review_path() -> Path:
    configured = os.environ.get("OPTILIFE_RAW_REVIEWS_CSV")
    if configured:
        return Path(configured).expanduser()
    return CANONICAL_RAW_REVIEWS_PATH


def resolve_labeled_review_path() -> Path:
    configured = os.environ.get("OPTILIFE_LABELED_REVIEWS_CSV")
    if configured:
        return Path(configured).expanduser()
    return CANONICAL_LABELED_REVIEWS_PATH


def csv_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": display_path(path),
        "exists": path.exists(),
        "row_count": 0,
        "columns": [],
    }
    if not path.exists():
        return info

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            info["columns"] = next(reader)
        except StopIteration:
            return info
        info["row_count"] = sum(1 for _ in reader)
    return info


def load_course_records() -> tuple[Path, List[Dict[str, Any]]]:
    path = resolve_course_path()
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Course CSV not found: {display_path(path)}",
        )

    review_path = resolve_labeled_review_path()
    review_input = review_path if review_path.exists() else None
    try:
        records = load_courses(path, review_input).to_dict("records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return path, [serialize_course(course) for course in records]


def clean_api_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def clean_api_string(value: Any) -> str:
    cleaned = clean_api_value(value)
    if cleaned is None:
        return ""
    return str(cleaned)


def serialize_course(course: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "course_id": clean_api_string(course.get("course_id", "")),
        "course_name": clean_api_string(course["course_name"]),
        "department": clean_api_string(course.get("department", "")),
        "section": clean_api_string(course.get("section", "")),
        "professor": clean_api_string(course.get("professor", "")),
        "category": clean_api_string(course.get("category", "")),
        "credits": int(course["credits"]),
        "core": bool(course["core"]),
        "difficulty_label": int(course["difficulty_label"]),
        "workload_label": int(course["workload_label"]),
        "grading_strictness_label": int(course.get("grading_strictness_label", 3)),
        "difficulty_average": float(course.get("difficulty_label_average", 3)),
        "workload_average": float(course.get("workload_label_average", 3)),
        "grading_strictness_average": float(
            course.get("grading_strictness_label_average", 3)
        ),
        "review_count": int(course.get("review_count", 0)),
        "label_source": clean_api_string(course.get("label_source", "")),
        "time_slot": clean_api_string(course["time_slot"]),
        "campus": clean_api_string(course.get("campus", "")),
        "classroom": clean_api_string(course.get("classroom", "")),
    }


def serialize_blocks(course: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = []
    for day, start, end in parse_slots(course["time_slot"]):
        blocks.append(
            {
                "day": day,
                "day_label": DAY_LABELS.get(day, day),
                "start": format_minutes(start),
                "end": format_minutes(end),
                "start_minutes": start,
                "end_minutes": end,
                "course_name": course["course_name"],
                "credits": course["credits"],
                "core": course["core"],
                "difficulty_label": course["difficulty_label"],
                "workload_label": course["workload_label"],
            }
        )
    return blocks


def serialize_recommendation(
    rank: int,
    score: float,
    schedule: List[Dict[str, Any]],
    reasons: List[str],
) -> Dict[str, Any]:
    courses = [serialize_course(course) for course in schedule]
    blocks = [
        block
        for course in courses
        for block in serialize_blocks(course)
    ]
    burden = daily_burden(courses)
    return {
        "rank": rank,
        "score": round(float(score), 2),
        "credits": sum(course["credits"] for course in courses),
        "difficulty_sum": sum(course["difficulty_label"] for course in courses),
        "workload_sum": sum(course["workload_label"] for course in courses),
        "reasons": reasons,
        "courses": courses,
        "blocks": blocks,
        "daily_burden": {
            DAY_LABELS.get(day, day): round(float(value), 2)
            for day, value in burden.items()
        },
    }


def condition_to_preferences(condition: Dict[str, Any]) -> Dict[str, Any]:
    preferences = {
        key: value
        for key, value in condition.items()
        if key not in {"min_credits", "max_credits", "limit"}
    }
    preferences["preferred_free_days"] = [
        normalize_day(day) or day
        for day in preferences.get("preferred_free_days", [])
    ]
    return preferences


def build_dataset_status() -> Dict[str, Any]:
    path, courses = load_course_records()
    raw_path = resolve_raw_review_path()
    labeled_path = resolve_labeled_review_path()
    courses_below_review_floor = [
        course["course_name"]
        for course in courses
        if course["review_count"] < 10
    ]
    return {
        "courses": csv_info(path),
        "raw_reviews": csv_info(raw_path),
        "labeled_reviews": csv_info(labeled_path),
        "course_count": len(courses),
        "review_floor": 10,
        "courses_below_review_floor": courses_below_review_floor,
        "ready": (
            bool(courses)
            and not courses_below_review_floor
            and labeled_path.exists()
        ),
    }


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        for header, value in NO_CACHE_HEADERS.items():
            response.headers[header] = value
    return response


def ui_asset_version() -> str:
    return str(
        max(path.stat().st_mtime_ns for path in UI_ASSET_PATHS if path.exists())
    )


def build_index_html() -> str:
    version = ui_asset_version()
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return (
        html
        .replace(
            'href="/static/styles.css"',
            f'href="/static/styles.css?v={version}"',
        )
        .replace(
            'src="/static/app.js"',
            f'src="/static/app.js?v={version}"',
        )
    )


@app.get("/", include_in_schema=False)
def index() -> Response:
    return Response(
        content=build_index_html(),
        headers=dict(NO_CACHE_HEADERS),
        media_type="text/html",
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    path, courses = load_course_records()
    using_sample = path == SAMPLE_COURSES_PATH
    dataset = build_dataset_status()
    return {
        "ok": True,
        "course_count": len(courses),
        "data_source": display_path(path),
        "review_data_source": display_path(resolve_labeled_review_path()),
        "data_status": "sample" if using_sample else "ready",
        "dataset": dataset,
        "flow": [
            "raw_everytime_reviews",
            "ai_labeling",
            "labeled_review_csv",
            "course_csv",
            "condition_ui",
            "schedule_recommendation",
        ],
    }


@app.get("/api/courses")
def courses() -> Dict[str, Any]:
    path, records = load_course_records()
    return {
        "data_source": display_path(path),
        "review_data_source": display_path(resolve_labeled_review_path()),
        "courses": records,
    }


@app.get("/api/admin/datasets")
def admin_datasets() -> Dict[str, Any]:
    return build_dataset_status()


@app.post("/api/admin/sync-reviews")
def sync_reviews() -> Dict[str, Any]:
    raw_path = resolve_raw_review_path()
    labeled_path = resolve_labeled_review_path()
    if not raw_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Raw review CSV not found: {display_path(raw_path)}",
        )

    LABELING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(LABELING_SCRIPT_PATH),
        "--input",
        str(raw_path),
        "--output",
        str(labeled_path),
        "--summary",
        str(LABELING_SUMMARY_PATH),
        "--merge-existing",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Review labeling script failed.",
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

    return {
        "ok": True,
        "message": "리뷰 동기화 완료",
        "stdout": result.stdout.strip(),
        "summary_path": display_path(LABELING_SUMMARY_PATH),
        "dataset": build_dataset_status(),
    }


@app.post("/api/recommend")
def recommend(condition_data: ConditionData) -> Dict[str, Any]:
    condition = model_to_dict(condition_data)
    if condition["min_credits"] > condition["max_credits"]:
        raise HTTPException(
            status_code=400,
            detail="min_credits must be less than or equal to max_credits.",
        )

    path, courses = load_course_records()
    preferences = condition_to_preferences(condition)
    recommendations = find_recommendations(
        courses,
        min_credits=condition["min_credits"],
        max_credits=condition["max_credits"],
        limit=condition["limit"],
        preferences=preferences,
    )

    return {
        "condition_data": condition,
        "data_source": display_path(path),
        "review_data_source": display_path(resolve_labeled_review_path()),
        "course_count": len(courses),
        "recommendations": [
            serialize_recommendation(rank, score, list(schedule), reasons)
            for rank, (score, schedule, reasons) in enumerate(
                recommendations,
                start=1,
            )
        ],
    }
