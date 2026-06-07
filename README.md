# OptiLife

조건 기반 시간표 추천 MVP입니다. 에브리타임 리뷰를 AI 모델이 라벨링한 CSV와 학교 포털 과목 목록 CSV를 결합해, UI에서 받은 조건에 맞는 시간표를 추천합니다.

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

## 데이터 흐름

```text
에브리타임 리뷰 CSV
-> AI 라벨링
-> data/csv/labeled_everytime_reviews.csv
-> data/csv/courses.csv와 조인
-> UI 조건 기반 시간표 추천
```

관리자 패널의 `리뷰 해석 및 동기화` 버튼은 `data/csv/raw_everytime_reviews.csv`를 입력으로 `scripts/label_everytime_reviews.py`를 실행하고, 결과를 `data/csv/labeled_everytime_reviews.csv`에 병합합니다.

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
