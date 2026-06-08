import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.recommend import (
    DAY_LABELS,
    build_recommendation_diagnostics,
    daily_burden,
    find_recommendations,
    format_minutes,
    load_courses,
    normalize_day,
    parse_slots,
    preference_warnings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
CANONICAL_COURSES_PATH = PROJECT_ROOT / "data" / "csv" / "courses.csv"
SAMPLE_COURSES_PATH = (
    PROJECT_ROOT / "scripts" / "examples" / "recommend" / "input" / "courses.csv"
)
COURSE_EVALUATION_COLUMNS = [
    "rating",
    "workload_label",
    "teamwork_load_label",
    "grading_strictness_label",
]

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
    core_min_count: int = Field(0, ge=0, le=30)
    core_max_count: int = Field(30, ge=0, le=30)
    categories: List[str] = Field(default_factory=list)
    preferred_free_days: List[str] = Field(default_factory=lambda: ["Fri"])
    avoid_early: bool = True
    avoid_friday_afternoon: bool = True
    balance_days: bool = True
    early_cutoff: str = "10:00"
    rating_weight: float = Field(1.0, ge=0, le=5)
    workload_weight: float = Field(1.1, ge=0, le=5)
    teamwork_weight: float = Field(0.8, ge=0, le=5)
    grading_weight: float = Field(0.9, ge=0, le=5)


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


def courses_missing_evaluation(path: Path) -> List[str]:
    if not path.exists():
        return []

    missing_courses = []
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            missing = [
                column
                for column in COURSE_EVALUATION_COLUMNS
                if clean_api_string(row.get(column, "")) == ""
            ]
            if missing:
                course_name = clean_api_string(row.get("course_name", ""))
                section = clean_api_string(row.get("section", ""))
                missing_courses.append(
                    f"{course_name}({section})" if section else course_name
                )
    return missing_courses


def load_course_records() -> tuple[Path, List[Dict[str, Any]]]:
    path = resolve_course_path()
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Course CSV not found: {display_path(path)}",
        )

    try:
        records = load_courses(path).to_dict("records")
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
        "academic_year": clean_api_string(course.get("academic_year", "")),
        "semester": clean_api_string(course.get("semester", "")),
        "course_name": clean_api_string(course["course_name"]),
        "department": clean_api_string(course.get("department", "")),
        "section": clean_api_string(course.get("section", "")),
        "professor": clean_api_string(course.get("professor", "")),
        "category": clean_api_string(course.get("category", "")),
        "credits": int(course["credits"]),
        "core": bool(course["core"]),
        "rating": float(course["rating"]),
        "workload_label": float(course["workload_label"]),
        "teamwork_load_label": float(course["teamwork_load_label"]),
        "grading_strictness_label": float(course["grading_strictness_label"]),
        "time_slot": clean_api_string(course["time_slot"]),
        "campus": clean_api_string(course.get("campus", "")),
        "classroom": clean_api_string(course.get("classroom", "")),
        "capacity": clean_api_string(course.get("capacity", "")),
        "source": clean_api_string(course.get("source", "")),
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
                "professor": course.get("professor", ""),
                "category": course.get("category", ""),
                "section": course.get("section", ""),
                "credits": course["credits"],
                "core": course["core"],
                "workload_label": course["workload_label"],
                "teamwork_load_label": course["teamwork_load_label"],
                "grading_strictness_label": course["grading_strictness_label"],
            }
        )
    return blocks


def serialize_recommendation(
    rank: int,
    score: float,
    schedule: List[Dict[str, Any]],
    reasons: List[str],
    preferences: Dict[str, Any],
) -> Dict[str, Any]:
    courses = [serialize_course(course) for course in schedule]
    blocks = [
        block
        for course in courses
        for block in serialize_blocks(course)
    ]
    burden = daily_burden(courses)
    category_counts: Dict[str, int] = {}
    for course in courses:
        category = course["category"] or "미분류"
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "rank": rank,
        "score": round(float(score), 2),
        "credits": sum(course["credits"] for course in courses),
        "core_count": sum(1 for course in courses if course["core"]),
        "category_counts": category_counts,
        "rating_sum": round(sum(course["rating"] for course in courses), 2),
        "workload_sum": round(sum(course["workload_label"] for course in courses), 2),
        "teamwork_sum": round(
            sum(course["teamwork_load_label"] for course in courses),
            2,
        ),
        "grading_sum": round(
            sum(course["grading_strictness_label"] for course in courses),
            2,
        ),
        "reasons": reasons,
        "unmet_preferences": preference_warnings(courses, preferences=preferences),
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
        if key
        not in {
            "min_credits",
            "max_credits",
            "limit",
            "core_min_count",
            "core_max_count",
            "categories",
        }
    }
    preferences["preferred_free_days"] = [
        normalize_day(day) or day
        for day in preferences.get("preferred_free_days", [])
    ]
    return preferences


def build_dataset_status() -> Dict[str, Any]:
    path, courses = load_course_records()
    missing_evaluation = courses_missing_evaluation(path)
    category_counts: Dict[str, int] = {}
    for course in courses:
        category = course["category"] or "미분류"
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "courses": csv_info(path),
        "course_count": len(courses),
        "core_course_count": sum(1 for course in courses if course["core"]),
        "categories": category_counts,
        "course_evaluation_columns": COURSE_EVALUATION_COLUMNS,
        "courses_missing_evaluation": missing_evaluation,
        "ready": bool(courses) and not missing_evaluation,
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
        "data_status": "sample" if using_sample else "ready",
        "dataset": dataset,
        "flow": [
            "courses_csv",
            "condition_ui",
            "schedule_recommendation",
        ],
    }


@app.get("/api/courses")
def courses() -> Dict[str, Any]:
    path, records = load_course_records()
    return {
        "data_source": display_path(path),
        "courses": records,
    }


@app.post("/api/recommend")
def recommend(condition_data: ConditionData) -> Dict[str, Any]:
    condition = model_to_dict(condition_data)
    if condition["min_credits"] > condition["max_credits"]:
        raise HTTPException(
            status_code=400,
            detail="min_credits must be less than or equal to max_credits.",
        )
    if condition["core_min_count"] > condition["core_max_count"]:
        raise HTTPException(
            status_code=400,
            detail="core_min_count must be less than or equal to core_max_count.",
        )

    path, courses = load_course_records()
    preferences = condition_to_preferences(condition)
    recommendations = find_recommendations(
        courses,
        min_credits=condition["min_credits"],
        max_credits=condition["max_credits"],
        limit=condition["limit"],
        preferences=preferences,
        core_min_count=condition["core_min_count"],
        core_max_count=condition["core_max_count"],
        categories=condition["categories"],
    )
    diagnostics = build_recommendation_diagnostics(
        courses,
        min_credits=condition["min_credits"],
        max_credits=condition["max_credits"],
        recommendations=recommendations,
        core_min_count=condition["core_min_count"],
        core_max_count=condition["core_max_count"],
        categories=condition["categories"],
    )

    return {
        "condition_data": condition,
        "data_source": display_path(path),
        "course_count": len(courses),
        "diagnostics": diagnostics,
        "recommendations": [
            serialize_recommendation(rank, score, list(schedule), reasons, preferences)
            for rank, (score, schedule, reasons) in enumerate(
                recommendations,
                start=1,
            )
        ],
    }
