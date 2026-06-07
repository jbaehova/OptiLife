# OptiLife

조건 기반 시간표 추천 MVP입니다. 자연어 문장을 별도 모델로 구조화하던 단계를 UI 입력으로 대체하고, 그 입력값을 `condition_data` JSON으로 만들어 추천 API에 전달합니다.

## 구조

- `app/main.py`: FastAPI 서버, 정적 UI 서빙, 추천 API
- `app/static/`: 순수 HTML/CSS/JS UI
- `scripts/recommend.py`: 시간표 조합 생성 및 점수 계산
- `scripts/extract_everytime_saved_html.py`: Everytime 저장 HTML/MHTML 리뷰 추출
- `scripts/label_everytime_reviews.py`: 리뷰 약지도 라벨링
- `scripts/train_difficulty_model.py`: TF-IDF 기반 난이도 모델 실험

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

## 과목 CSV

기본 서버는 `data/csv/courses.csv`가 있으면 그 파일을 사용하고, 없으면 `scripts/examples/recommend/input/courses.csv` 샘플을 사용합니다. 다른 파일을 쓰려면 서버 실행 전에 환경변수를 지정합니다.

```bash
OPTILIFE_COURSES_CSV=/path/to/courses.csv python3 -m uvicorn app.main:app --reload
```

필수 컬럼은 아래와 같습니다.

```csv
course_name,credits,core,difficulty_label,workload_label,time_slot
자료구조,3,true,4,4,Mon 09:00-10:30;Wed 09:00-10:30
```

`time_slot`은 `Mon|Tue|Wed|Thu|Fri HH:MM-HH:MM` 형식을 세미콜론으로 연결합니다.
