# 서울 행정동 유망 업종 추천 (성장 vs 폐업위험)

서울시 **행정동(dong_code)** 단위로 업종별 **매출 성장(기회)** 과 **폐업 위험(리스크)** 를 동시에 고려해,
유망 업종을 추천해주는 간단한 추천 서비스입니다.

- 성장 모델: 다음 분기 성장(상위 20%) 확률 예측
- 위험 모델: 다음 분기 폐업위험(상위 20%) 확률 예측
- 추천 점수: `score = proba_growth - alpha * proba_risk`
  - `alpha`가 클수록 안정형(리스크 낮은) 추천에 가중치

---

## 실행 방법 (로컬)

### 1) 설치
가상환경(권장) 후 설치: pip install -r requirements.txt

### 2) API 서버 실행
프로젝트 폴더( app.py 있는 위치 )에서 실행: uvicorn app:app --reload --host 0.0.0.0 --port 8000

정상 실행 확인: http://localhost:8000/health

### 3) 프론트(index.html) 실행
주의: python -m http.server는 반드시 index.html이 있는 폴더에서 실행해야 404가 안 납니다.

프로젝트 폴더에서: python -m http.server 5500

브라우저에서 열기: http://localhost:5500/index.html

index.html 안의 const API = "http://localhost:8000"; 값이 API 서버 주소입니다.
포트를 바꿨다면 이 값도 맞춰주세요.

## 사용 방법
1. 분기(quarter) 선택 (예: 2024Q3)

2. 검색창에 행정동 코드(예: 11740620) 또는 동 이름(예: 대치) 입력

3. 검색 결과에서 행정동 선택 → 추천 결과 출력

4. alpha 값을 조절해서 안정형/공격형 추천을 비교 가능

## API 엔드포인트
- GET /health : 헬스 체크

- GET /quarters : 사용 가능한 분기 목록

- GET /search_dong?q=...&quarter=...

    - 행정동 코드/이름으로 검색 (dong_map.csv가 있으면 이름 검색 가능)

- GET /recommend?dong_code=...&quarter=...&top_n=10&alpha=1.0

    - 추천 결과 반환

    - 응답에 proba_growth, proba_risk, score, reasons(간단한 추천 이유) 포함

## 노트북(코랩) 구성
01_데이터수집_전처리.ipynb

    - 원천 데이터 로드/정리 → 패널 생성

    - 산출물: features_panel.parquet, biz_code_map.csv, dong_map.csv

02_모델링_검증.ipynb

    - 성장/위험 모델 학습 및 평가

    - 산출물: model_growth.pkl, model_risk.pkl

03_추천시스템_데모.ipynb

    - 저장된 산출물 로드 후 추천 데모 실행

## 한계 및 개선 아이디어

생활인구(pop_*) 등 일부 피처는 데이터 기간이 맞지 않으면 결측이 많아져 모델에 반영되지 않을 수 있습니다.

    - 개선: 분기(2024Q1~Q3)에 맞는 생활인구 데이터로 정합성 확보

모델 저장(pkl)은 학습/서빙 환경의 scikit-learn 버전이 다르면 경고/오류가 날 수 있습니다.

    - 개선: 학습/서빙 환경 버전 통일 또는 재학습
