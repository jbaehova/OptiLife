# OptiLife

조건 기반 시간표 추천 MVP입니다. 에브리타임 리뷰를 AI 모델이 라벨링한 CSV와 학교 포털 과목 목록 CSV를 결합해, UI에서 받은 조건에 맞는 시간표를 추천합니다.

현재 저장소에는 팀 개발을 위한 mock 데이터가 포함되어 있습니다.

- `data/csv/courses.csv`: 과목 목록 mock 12개
- `data/csv/raw_everytime_reviews.csv`: 라벨링 전 mock 리뷰 100개
- `data/csv/labeled_everytime_reviews.csv`: 기존 자료구조개론/알고리즘개론 라벨 리뷰 + mock 라벨 리뷰

## 구조

- `app/main.py`: FastAPI 서버, 정적 UI 서빙, 추천 API
- `app/static/`: 순수 HTML/CSS/JS UI
- `scripts/recommend.py`: 시간표 조합 생성 및 점수 계산
- `scripts/extract_everytime_saved_html.py`: Everytime 저장 HTML/MHTML 리뷰 추출
- `scripts/label_everytime_reviews.py`: 리뷰 약지도 라벨링
- `scripts/train_difficulty_model.py`: TF-IDF 기반 난이도 모델 실험
- `docs/data_contract.md`: raw review, labeled review, course catalog CSV 계약

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
data/csv/raw_everytime_reviews.csv
-> AI 라벨링
-> data/csv/labeled_everytime_reviews.csv
-> data/csv/courses.csv와 조인
-> UI 조건 기반 시간표 추천
```

관리자 패널의 `리뷰 해석 및 동기화` 버튼은 `data/csv/raw_everytime_reviews.csv`를 입력으로 `scripts/label_everytime_reviews.py`를 실행하고, 결과를 `data/csv/labeled_everytime_reviews.csv`에 병합합니다.

기존 labeled CSV에만 있는 리뷰는 삭제하지 않습니다. 같은 `review_id`가 있으면 갱신하고, 새 리뷰는 뒤에 추가합니다.

## CSV 계약

기본 서버는 `data/csv/courses.csv`, `data/csv/raw_everytime_reviews.csv`, `data/csv/labeled_everytime_reviews.csv`를 사용합니다. 세부 컬럼 계약은 [docs/data_contract.md](docs/data_contract.md)를 기준으로 합니다.

다른 파일을 쓰려면 서버 실행 전에 환경변수를 지정합니다.

```bash
OPTILIFE_COURSES_CSV=/path/to/courses.csv \
OPTILIFE_RAW_REVIEWS_CSV=/path/to/raw_everytime_reviews.csv \
OPTILIFE_LABELED_REVIEWS_CSV=/path/to/labeled_everytime_reviews.csv \
python3 -m uvicorn app.main:app --reload
```

과목 CSV의 추천 필수 컬럼은 아래와 같습니다.

```csv
course_name,credits,core,time_slot
자료구조개론,3,true,Mon 09:00-10:30;Wed 09:00-10:30
```

`time_slot`은 `Mon|Tue|Wed|Thu|Fri HH:MM-HH:MM` 형식을 세미콜론으로 연결합니다.

## 주요 API

- `GET /api/health`: 앱 상태, CSV 경로, row count, 과목별 라벨 리뷰 충족 여부
- `GET /api/courses`: 과목 목록과 labeled review 집계 레이블
- `POST /api/recommend`: UI 조건을 기반으로 시간표 추천
- `GET /api/admin/datasets`: 관리자 데이터 상태
- `POST /api/admin/sync-reviews`: raw review CSV를 라벨링해서 labeled review CSV에 병합

## CLI 사용

추천 로직만 확인하려면 아래 명령을 실행합니다.

```bash
python3 scripts/recommend.py \
  --data data/csv/courses.csv \
  --reviews data/csv/labeled_everytime_reviews.csv \
  --output outputs/recommendations.txt
```

리뷰 라벨링만 실행하려면 아래 명령을 사용합니다.

```bash
python3 scripts/label_everytime_reviews.py \
  --input data/csv/raw_everytime_reviews.csv \
  --output data/csv/labeled_everytime_reviews.csv \
  --summary outputs/review_labeling_summary.txt \
  --merge-existing
```

`outputs/` 아래 파일은 생성물입니다. 삭제해도 다음 실행 때 다시 만들어지며, git에는 포함하지 않습니다.

## 검증

현재 별도 테스트 suite는 없습니다. 변경 후 최소한 아래 명령을 실행합니다.

```bash
python3 -m compileall app scripts
```

서버 실행 후 smoke test 예시는 아래와 같습니다.

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s -X POST http://127.0.0.1:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"min_credits":15,"max_credits":18,"limit":3}'
```
