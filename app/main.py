import os
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
SAMPLE_COURSES_PATH = (
    PROJECT_ROOT / "scripts" / "examples" / "recommend" / "input" / "courses.csv"
)

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


def serialize_course(course: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "course_name": str(course["course_name"]),
        "credits": int(course["credits"]),
        "core": bool(course["core"]),
        "difficulty_label": int(course["difficulty_label"]),
        "workload_label": int(course["workload_label"]),
        "time_slot": str(course["time_slot"]),
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
    return {
        "ok": True,
        "course_count": len(courses),
        "data_source": display_path(path),
        "data_status": "sample" if using_sample else "ready",
        "flow": [
            "condition_ui",
            "condition_data",
            "course_csv",
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
        "course_count": len(courses),
        "recommendations": [
            serialize_recommendation(rank, score, list(schedule), reasons)
            for rank, (score, schedule, reasons) in enumerate(
                recommendations,
                start=1,
            )
        ],
    }
