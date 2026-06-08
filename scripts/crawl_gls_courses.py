"""
킹고정보시스템(kingoinfo.skku.edu) 수강편람 크롤러
Playwright로 브라우저를 자동화하여 GAIA SSV 응답을 가로채고
data/csv/courses.csv 형식으로 저장합니다.

최초 1회 브라우저 설치 필요:
    python -m playwright install chromium

실행:
    python scripts/crawl_gls_courses.py --year 2026 --semester 1 --user-id 학번 --password 비밀번호

브라우저 창을 직접 보면서 실행하려면:
    python scripts/crawl_gls_courses.py --year 2026 --semester 1 --user-id 학번 --password 비밀번호 --headed
"""

import argparse
import csv
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# --- 설정 ---
# ---------------------------------------------------------------------------

KINGO_URL   = "https://kingoinfo.skku.edu/gaia/nxui/index.html"
LOGIN_URL   = "https://icampus.skku.edu/xn-sso/login.php?auto_login=true&sso_only=true&cvs_lgn="

# 수강편람 API endpoint (Network 탭에서 확인된 URL)
# 전공: NHSSU030530M/selectMain.do
# 교양: NHSSU030540M/selectMain03.do
COURSE_API_PATHS = {
    "/gaia/NHSSU030530M/selectMain.do",
    "/gaia/NHSSU030540M/selectMain03.do",
}

# GAESUL_TERM 인코딩: 1학기="10", 2학기="20"
SEMESTER_CODE = {1: "10", 2: "20"}

SOURCE        = "gls"

# 페이지 로드 최대 대기 시간 (ms)
TIMEOUT_MS = 60_000

# ---------------------------------------------------------------------------
# --- 필드 정의 ---
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "course_id", "academic_year", "semester", "department",
    "course_name", "section", "professor", "credits", "core",
    "category", "time_slot", "campus", "classroom", "capacity", "source",
]

EVALUATION_COLUMNS = [
    "rating", "workload_label", "teamwork_load_label", "grading_strictness_label",
]

ALL_FIELDNAMES = FIELDNAMES + EVALUATION_COLUMNS

CORE_KEYWORDS = {"전공필수", "교양필수", "전공코어"}

# ---------------------------------------------------------------------------
# --- 요일 변환 ---
# ---------------------------------------------------------------------------

DAY_MAP   = {"월": "Mon", "화": "Tue", "수": "Wed", "목": "Thu", "금": "Fri"}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ---------------------------------------------------------------------------
# --- GAIA SSV 파서 ---
# ---------------------------------------------------------------------------
# 킹고정보시스템 실제 응답 형식:
#   SSV:UTF-8
#   ErrorCode:int=0
#   ErrorMsg:string=SUCCESS
#   Dataset:dsGrdMain
#   _RowType_\x1f_chk:string(32)\x1fGAESUL_YEAR:string(4)\x1f...   ← 컬럼 정의 (구분자 \x1f)
#   N\x1f0\x1f2026\x1f10\x1f자연과학\x1f...                         ← 데이터 행
#
# 구분자: \x1f (Unit Separator, ASCII 0x1F) - 탭이 아님
# ---------------------------------------------------------------------------

SEP = "\x1f"  # GAIA SSV 실제 구분자


def _parse_field_name(col_def: str) -> str:
    """컬럼 정의 문자열에서 필드 이름만 추출합니다.
    예: "GAESUL_YEAR:string(4)" → "GAESUL_YEAR"
        "_chk:string(32)"       → "_chk"
    """
    return col_def.split(":")[0].strip()


_HEADER_FIELD_RE = re.compile(
    r'^[A-Za-z_]\w*(?::[a-z]+\(\d+\))?$'
)

def _is_ssv_header(line: str) -> bool:
    """GAIA SSV 컬럼 정의 행인지 판별합니다.

    진짜 헤더는 각 토큰이 'FIELDNAME' 또는 'FIELDNAME:type(size)' 형태입니다.
    예) '_RowType_\x1f_chk:string(32)\x1fGAESUL_YEAR:string(4)...'
    데이터 분할로 생긴 가짜 헤더(예: '플립러닝(온라인...)') 는 걸러냅니다.
    """
    if not line or SEP not in line:
        return False
    if line.startswith("N" + SEP):
        return False
    tokens = line.split(SEP)
    # 첫 토큰이 영문/언더스코어로 시작하는 식별자여야 진짜 헤더
    return bool(_HEADER_FIELD_RE.match(tokens[0]))


def parse_gaia_ssv(text: str) -> list[dict]:
    """GAIA SSV 응답 텍스트를 dict 리스트로 파싱합니다.

    - 헤더에서 필드 이름을 동적으로 읽습니다.
    - INFORM/BIGO 등 필드에 개행이 포함된 행을 올바르게 이어 붙입니다.
      · 새 데이터 행은 반드시 'N<SEP>' 로 시작
      · 새 헤더 행은 영문 식별자:type(size) 패턴으로 시작
      · 그 외 줄은 이전 행의 연속으로 취급
    """
    fields: list[str] = []
    records: list[dict] = []
    pending: str = ""

    def flush(raw_line: str) -> None:
        if not fields or not raw_line:
            return
        parts = raw_line.split(SEP)
        while len(parts) < len(fields):
            parts.append("")
        records.append(dict(zip(fields, parts)))

    for line in text.splitlines():
        line = line.rstrip("\r")

        if _is_ssv_header(line):
            # 진짜 컬럼 정의 행 → 대기 중인 행을 먼저 flush
            if pending:
                flush(pending)
                pending = ""
            raw_fields = line.split(SEP)
            fields = [_parse_field_name(f) for f in raw_fields]

        elif line.startswith("N") and SEP in line:
            # 새 데이터 행 시작
            if pending:
                flush(pending)
            pending = line

        elif pending:
            # INFORM/BIGO 등에 포함된 개행 → 이전 행에 이어 붙임
            pending += " " + line.strip()

        # else: 메타 줄 (SSV:UTF-8, ErrorCode:int=0 등) → 무시

    if pending:
        flush(pending)

    return records

# ---------------------------------------------------------------------------
# --- 시간 슬롯 파싱 및 병합 ---
# ---------------------------------------------------------------------------
# 입력: "월18:00-18:50【26310】,월19:00-19:50【26310】,월20:00-20:50【26310】"
# 출력: "Mon 18:00-20:50"
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(
    r"([월화수목금])"
    r"(\d{1,2}:\d{2})"
    r"[-~]"
    r"(\d{1,2}:\d{2})"
    r"(?:【[^】]*】)?"
)


def _to_min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _from_min(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def parse_skku_time_slot(gyosi_name: str) -> str:
    """SKKU GYOSI_NAME → courses.csv time_slot 형식으로 변환합니다.

    연속된 50분 슬롯을 하나의 블록으로 병합합니다 (10분 이내 간격).
    반환: "Mon 18:00-21:45;Wed 09:00-10:30"
    """
    if not gyosi_name or not gyosi_name.strip():
        return ""

    raw: list[tuple[str, int, int]] = []
    for m in _SLOT_RE.finditer(gyosi_name):
        eng = DAY_MAP.get(m.group(1))
        if eng:
            raw.append((eng, _to_min(m.group(2)), _to_min(m.group(3))))

    if not raw:
        return gyosi_name

    raw.sort(key=lambda s: (DAY_ORDER.index(s[0]), s[1]))

    merged: list[tuple[str, int, int]] = []
    for day, start, end in raw:
        if merged and merged[-1][0] == day and start - merged[-1][2] <= 10:
            d, s, _ = merged[-1]
            merged[-1] = (d, s, max(merged[-1][2], end))
        else:
            merged.append((day, start, end))

    return ";".join(
        f"{d} {_from_min(s)}-{_from_min(e)}" for d, s, e in merged
    )

# ---------------------------------------------------------------------------
# --- 학점 / 학기 / core 변환 ---
# ---------------------------------------------------------------------------

def parse_credits(hakjum: str) -> int:
    """"3(3)" → 3,  "2(4)" → 2"""
    m = re.match(r"(\d+)", str(hakjum).strip())
    return int(m.group(1)) if m else 0


def term_to_semester(gaesul_term: str) -> int:
    return {"10": 1, "20": 2, "1": 1, "2": 2}.get(str(gaesul_term).strip(), 0)


def to_core(isu_name: str, isu2: str = "", isu3: str = "") -> bool:
    combined = f"{isu_name} {isu2} {isu3}"
    return any(kw in combined for kw in CORE_KEYWORDS)

# ---------------------------------------------------------------------------
# --- 과목 정규화 ---
# ---------------------------------------------------------------------------

def normalize_course(raw: dict) -> dict | None:
    course_name = raw.get("GWAMOK_NAME", "").strip()

    # 한국어 과목명이 없으면 영문명으로 대체
    if not course_name:
        course_name = raw.get("GWAMOK_ENG_NAME", "").strip()

    # 과목명 + 과목코드 모두 없으면 스킵
    course_id_raw = raw.get("HAKSU_NO", "").strip()
    if not course_name and not course_id_raw:
        return None

    isu_name  = raw.get("ISU_NAME",  "").strip()
    isu_name2 = raw.get("ISU_NAME2", "").strip()
    isu_name3 = raw.get("ISU_NAME3", "").strip()
    gyosi     = raw.get("GYOSI_NAME", "").strip()

    classroom_m = re.search(r"【([^】]+)】", gyosi)
    classroom = classroom_m.group(1) if classroom_m else ""
    if classroom in {"미지정", ""}:
        classroom = ""

    dept_m = re.match(r"[A-Za-z]+", course_id_raw)
    return {
        "course_id":     course_id_raw,
        "academic_year": int(raw.get("GAESUL_YEAR", 0) or 0),
        "semester":      term_to_semester(raw.get("GAESUL_TERM", "")),
        "department":    dept_m.group() if dept_m else "",
        "course_name":   course_name,
        "section":       raw.get("BUNBAN", "").strip(),
        "professor":     raw.get("PER_NAME", "").strip(),
        "credits":       parse_credits(raw.get("HAKJUM", "")),
        "core":          str(to_core(isu_name, isu_name2, isu_name3)).lower(),
        "category":      isu_name or isu_name2 or isu_name3,
        "time_slot":     parse_skku_time_slot(gyosi),
        "campus":        raw.get("CAMPUS_NM", "").strip(),
        "classroom":     classroom,
        "capacity":      "",
        "source":        SOURCE,
    }

# ---------------------------------------------------------------------------
# --- Playwright 브라우저 자동화 ---
# ---------------------------------------------------------------------------

def fetch_ssv_with_playwright(
    user_id: str,
    password: str,
    year: int,
    semester: int,
    headed: bool = False,
) -> list[str]:
    """Playwright로 킹고정보시스템 수강편람 SSV 데이터를 캡처합니다.

    브라우저를 열어두고 여러 번 조회할 수 있습니다.
    터미널에서 Enter를 누르면 수집을 종료하고 저장합니다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "오류: playwright가 설치되지 않았습니다.\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        )
        sys.exit(1)

    ssv_results: list[str] = []
    capture_count = [0]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # ── 수강편람 API 응답 가로채기 (조회할 때마다 누적) ─────────────────
        def on_response(response):
            if any(path in response.url for path in COURSE_API_PATHS):
                try:
                    text = response.text()
                    if text.startswith("SSV:"):
                        capture_count[0] += 1
                        ssv_results.append(text)
                        # 원본 SSV 즉시 저장
                        debug_dir = "outputs/debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        path = f"{debug_dir}/raw_ssv_{capture_count[0]:02d}.txt"
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(text)
                        print(
                            f"\n[{capture_count[0]}번째 캡처] "
                            f"{len(text):,} bytes → {path}"
                        )
                        print("계속 다른 조건으로 조회하거나, "
                              "터미널에서 Enter를 눌러 저장하세요.")
                except Exception:
                    pass

        page.on("response", on_response)

        # ── 킹고정보시스템 열기 ──────────────────────────────────────────────
        page.goto(KINGO_URL, timeout=TIMEOUT_MS, wait_until="domcontentloaded")

        print("\n" + "=" * 60)
        print("브라우저가 열렸습니다.")
        print()
        print("  1. 로그인")
        print("  2. 수강편람 → 조건 선택 → 조회  (여러 번 반복 가능)")
        print("  3. 다 조회했으면 터미널에서 Enter 키를 누르세요")
        print("=" * 60)

        # ── 터미널 Enter 입력 대기 (백그라운드 스레드) ──────────────────────
        import threading
        done = threading.Event()

        def wait_for_enter():
            input("\n모든 조회가 끝나면 Enter를 누르세요... ")
            done.set()

        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()

        # ── Enter가 눌릴 때까지 브라우저 유지 ───────────────────────────────
        while not done.is_set():
            try:
                page.wait_for_timeout(1000)
            except Exception:
                break

        print(f"\n총 {capture_count[0]}번 캡처됨. 브라우저를 닫습니다...")
        try:
            browser.close()
        except Exception:
            pass

    if not ssv_results:
        print("오류: 수강편람 응답을 캡처하지 못했습니다.")
        sys.exit(1)

    return ssv_results

# ---------------------------------------------------------------------------
# --- CSV 저장 / 병합 ---
# ---------------------------------------------------------------------------

def _course_key(row: dict) -> tuple:
    return (
        str(row.get("course_id", "")).strip().upper(),
        str(row.get("section", "")).strip(),
        str(row.get("academic_year", "")),
        str(row.get("semester", "")),
    )


def load_existing_courses(path: str) -> dict[tuple, dict]:
    """기존 courses.csv를 읽어 key → row 딕셔너리로 반환합니다."""
    existing: dict[tuple, dict] = {}
    if not os.path.exists(path):
        return existing
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            existing[_course_key(row)] = row
    return existing


def merge_into_existing(
    existing: dict[tuple, dict],
    new_courses: list[dict],
) -> list[dict]:
    """기존 과목에 새 과목을 병합합니다.

    - 같은 키(course_id + section + year + semester)가 있으면
      기본 필드만 갱신하고 평가 컬럼(rating 등)은 기존 값을 보존합니다.
    - 새 과목은 평가 컬럼을 빈값으로 추가합니다.
    """
    merged = {k: dict(v) for k, v in existing.items()}
    added = updated = 0

    for course in new_courses:
        key = _course_key(course)
        if key in merged:
            for field in FIELDNAMES:
                merged[key][field] = course.get(field, merged[key].get(field, ""))
            updated += 1
        else:
            row = dict(course)
            for col in EVALUATION_COLUMNS:
                row.setdefault(col, "")
            merged[key] = row
            added += 1

    print(f"병합 결과: {added}개 추가, {updated}개 갱신, 기존 유지 {len(existing) - updated}개 → 최종 {len(merged)}개")
    return list(merged.values())


def save_courses(courses: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(courses)
    print(f"{len(courses)}개 과목을 '{output_path}'에 저장했습니다.")

# ---------------------------------------------------------------------------
# --- 메인 ---
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="킹고정보시스템 수강편람을 크롤링하여 courses.csv로 저장합니다."
    )
    parser.add_argument("--year",     type=int, required=True, help="개설 연도 (예: 2026)")
    parser.add_argument("--semester", type=int, required=True, choices=[1, 2],
                        help="개설 학기 (1 또는 2)")
    parser.add_argument("--user-id",  default="",
                        help=f"로그인 학번. 환경변수 {ENV_USER_ID} 가 우선 적용됩니다.")
    parser.add_argument("--password", default="",
                        help=f"로그인 비밀번호. 환경변수 {ENV_PASSWORD} 가 우선 적용됩니다.")
    parser.add_argument("--output",   default="data/csv/courses.csv",
                        help="저장할 CSV 경로 (기본값: data/csv/courses.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 CSV를 병합하지 않고 완전히 덮어씁니다")
    parser.add_argument("--headed", action="store_true",
                        help="(무시됨: 항상 브라우저 창이 열립니다)")
    parser.add_argument("--from-ssv", default="",
                        help="브라우저 없이 저장된 SSV 파일에서 직접 파싱. "
                             "쉼표로 여러 파일 지정 가능 "
                             "(예: outputs/debug/raw_ssv_01.txt,outputs/debug/raw_ssv_02.txt)")
    args = parser.parse_args()

    user_id  = args.user_id
    password = args.password

    if not user_id or not password:
        print(
            "오류: 로그인 정보가 없습니다.\n"
            "CLI 인자로 학번과 비밀번호를 입력하세요.\n"
            "  --user-id 학번 --password 비밀번호"
        )
        sys.exit(1)

    # 저장된 SSV 파일이 있으면 브라우저 없이 바로 파싱
    if args.from_ssv:
        ssv_texts = []
        for path in args.from_ssv.split(","):
            path = path.strip()
            with open(path, "r", encoding="utf-8") as f:
                ssv_texts.append(f.read())
            print(f"저장된 SSV 파일 로드: {path}")
    else:
        # Playwright로 SSV 데이터 수집 (브라우저 창 열림 → 수동 로그인 + 반복 조회)
        ssv_texts = fetch_ssv_with_playwright(
            user_id=user_id,
            password=password,
            year=args.year,
            semester=args.semester,
        )

    # 모든 SSV 파싱 후 합치기
    all_raw: list[dict] = []
    for ssv_text in ssv_texts:
        all_raw.extend(parse_gaia_ssv(ssv_text))
    print(f"\n전체 파싱된 레코드: {len(all_raw)}개 ({len(ssv_texts)}번 조회)")

    # 파싱 결과 디버깅: 첫 번째 레코드 출력
    if all_raw:
        print("\n[디버그] 첫 번째 레코드 필드:")
        for k, v in all_raw[0].items():
            if v.strip():
                print(f"  {k}: {repr(v)}")
        print()

    # courses.csv 스키마로 변환 + 중복 제거
    # 중복 기준: course_id + section + academic_year + semester
    # (공백·대소문자 정규화 후 비교)
    courses_dict: dict[tuple, dict] = {}
    skipped_rows: list[dict] = []
    dup_count = 0

    for raw in all_raw:
        course = normalize_course(raw)
        if course is None:
            skipped_rows.append(raw)
            continue
        key = (
            str(course["course_id"]).strip().upper(),
            str(course["section"]).strip(),
            str(course["academic_year"]),
            str(course["semester"]),
        )
        if key in courses_dict:
            dup_count += 1
        else:
            courses_dict[key] = course

    courses = list(courses_dict.values())

    if skipped_rows:
        print(f"\n{len(skipped_rows)}개 레코드 스킵됨 (과목명·과목코드 모두 없음):")
        for r in skipped_rows:
            fields_preview = {k: v for k, v in r.items() if v.strip()}
            print(f"  {fields_preview}")
        # 스킵된 항목을 별도 파일에 저장 (검토용)
        skip_path = os.path.join(os.path.dirname(args.output), "skipped_records.csv")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(skip_path, "w", newline="", encoding="utf-8-sig") as f:
            all_keys = list(dict.fromkeys(k for r in skipped_rows for k in r))
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(skipped_rows)
        print(f"  → {skip_path} 에 저장했습니다 (수동 검토 가능)")
    if dup_count:
        print(f"{dup_count}개 중복 제거됨")
    print(f"최종 과목 수: {len(courses)}개")

    if not courses:
        print("\n과목명을 찾지 못했습니다.")
        print("outputs/debug/ 폴더의 raw_ssv_*.txt 파일을 확인하세요.")
        sys.exit(1)

    if args.overwrite:
        final_courses = courses
    else:
        existing = load_existing_courses(args.output)
        final_courses = merge_into_existing(existing, courses)

    save_courses(final_courses, args.output)


if __name__ == "__main__":
    main()
