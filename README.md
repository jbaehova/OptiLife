# OptiLife

애플리케이션 레이어는 `data/csv/courses.csv`만 읽습니다. 리뷰 라벨링과 모델 실험 스크립트는 팀원이 오프라인으로 실행해 `courses.csv`의 course-level 평가 컬럼을 준비하는 용도입니다.

- `data/csv/courses.csv`: Course 목록과 course-level 평가 컬럼

## 구조

- `app/main.py`: FastAPI 서버, 정적 UI 서빙, 추천 API
- `app/static/`: 순수 HTML/CSS/JS UI
- `scripts/recommend.py`: 시간표 조합 생성 및 점수 계산
- `scripts/extract_everytime_saved_html.py`: 오프라인 Everytime 저장 HTML/MHTML 리뷰 추출
- `scripts/label_everytime_reviews.py`: 오프라인 리뷰 약지도 라벨링
- `scripts/train_difficulty_model.py`: 오프라인 TF-IDF 기반 리뷰 라벨 모델 실험
- `docs/data_contract.md`: course catalog CSV 계약과 오프라인 리뷰 CSV 계약

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

`uv`를 쓰는 경우에는 아래처럼 설치할 수 있습니다.

```bash
uv sync
python3 -m uvicorn app.main:app --reload
```

## 데이터 흐름

```text
data/csv/courses.csv
-> UI 조건 기반 시간표 추천
```

추천 점수는 `courses.csv`의 `rating`, `workload_label`, `teamwork_load_label`, `grading_strictness_label`을 사용합니다. 전공필수는 가중치가 아니라 최소/최대 개수 조건으로 처리합니다.

## CSV 계약

기본 서버는 `data/csv/courses.csv`를 사용합니다. 세부 컬럼 계약은 [docs/data_contract.md](docs/data_contract.md)를 기준으로 합니다.

다른 파일을 쓰려면 서버 실행 전에 환경변수를 지정합니다.

```bash
OPTILIFE_COURSES_CSV=/path/to/courses.csv python3 -m uvicorn app.main:app --reload
```

과목 CSV의 추천 필수 컬럼은 아래와 같습니다.

```csv
course_name,credits,core,rating,workload_label,teamwork_load_label,grading_strictness_label,time_slot
자료구조개론,3,true,4.2,2.4,4.7,3.1,Mon 09:00-10:30;Wed 09:00-10:30
```

`time_slot`은 `Mon|Tue|Wed|Thu|Fri HH:MM-HH:MM` 형식을 세미콜론으로 연결합니다. 에브리타임 3단계 평가는 `없음/너그러움=5`, `보통=3`, `많음/깐깐함=1`로 환산한 가중평균 숫자입니다.

## 주요 API

- `GET /api/health`: 앱 상태, CSV 경로, row count, course-level 평가 컬럼 준비 여부
- `GET /api/courses`: 과목 목록과 에브리타임 평가 지표
- `POST /api/recommend`: UI 조건을 기반으로 시간표 추천과 실패 진단 반환

## CLI 사용

추천 로직만 확인하려면 아래 명령을 실행합니다.

```bash
python3 scripts/recommend.py \
  --data data/csv/courses.csv \
  --output outputs/recommendations.txt
```

오프라인 리뷰 라벨링은 앱 서버와 분리해서 실행합니다.

```bash
python3 scripts/label_everytime_reviews.py \
  --input data/csv/raw_everytime_reviews.csv \
  --output data/csv/labeled_everytime_reviews.csv \
  --summary outputs/review_labeling_summary.txt \
  --merge-existing
```

`outputs/` 아래 파일은 생성물입니다. 삭제해도 다음 실행 때 다시 만들어지며, git에는 포함하지 않습니다.

## 검증

```bash
python3 -m compileall app scripts
python3 -m pytest
```