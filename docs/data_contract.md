### 아키텍쳐 

리뷰 데이터 csv (raw_everytime_reviews.csv) -> ai 모델이 레이블 달음 -> 이걸 어딘가에 저장 (/data/csv/labeled_everytime_reviews.csv 와 같은 형태로) -> 유저 input 받음 (UI 로 받음, 조건들) -> 기존에 ai 가 레이블 달은 정보를 기반으로 시간표 추천



### 데이터 CSV

CSV 는 다음 세개만 사용합니다.

1. data/csv/raw_everytime_reviews.csv
2. data/csv/labeled_everytime_reviews.csv
3. data/csv/courses.csv

**1. raw_everytime_reviews.csv**
라벨링 전 에브리타임 리뷰 원문입니다. 저희가 에타에서 데이터 긁어와서 저장하는 곳입니다.

스키마
- review_id: 리뷰 단위 고유 ID
- source_url: 에브리타임 강의 URL
- lecture_id: 에브리타임 강의 ID
- course_name: 과목명
- professor: 교수명 
- semester: 리뷰 작성자가 표시한 수강 학기 문자열
- rating: 1-5 별점
- raw_review_text: 라벨링 모델 입력 텍스트

**2. labeled_everytime_reviews.csv**
1번을 ai 모델로 라벨링 한 뒤에 저장되는 곳입니다. 

스키마
- review_id
- source_url
- lecture_id
- course_name
- professor
- semester
- rating
- raw_review_text
- difficulty_label : 1 쉬움, 3 보통, 5 어려움.
- workload_label : 1 적음, 3 보통, 5 많음.
- grading_strictness_label : 1 후함, 3 보통, 5 엄격함.

**3. courses.csv**

GLS 등에서 크롤링 해서 채우는 "과목 카탈로그" 입니다. 이걸 시간표 추천할 때 사용합니다. 
유저 input 에 있는 여러 조건들에 대한 일치 여부가 여기에 포함되어있습니다.

스키마
1. 필수 컬럼 :
- course_name
- credits
- core
- time_slot : 형식 -> "Mon 09:00-10:30;Wed 09:00-10:30"

2. 권장 컬럼 :
- course_id : 포털 또는 내부 과목 고유 ID
- academic_year : 개설 연도
- semester : 개설 학기
- department : 개설 학과
- section : 분반
- professor : 교수명
- category : 전공필수, 전공선택, 교양필수 등 한글로 들어가있음 (core 와 비슷합니다)
- campus : UI 표시용 메타데이터
- classroom : UI 표시용 메타데이터
- capacity : UI 표시용 메타데이터
- source: 데이터 출처 표시용
