# ⚡ 설비 고장정보 조회 시스템

AI 기반 자연어 질의로 설비 고장 데이터를 조회하는 Streamlit 앱입니다.

---

## 📁 폴더 구조

```
failure_system/
│
├── Home.py                         ← Streamlit 진입점 (streamlit run Home.py)
│
├── pages/                          ← Streamlit 멀티페이지
│   ├── 1_자연어_조회.py             ← 자연어 → AI → 결과 테이블
│   └── 2_현황_대시보드.py           ← 집계 차트 대시보드
│
├── utils/                          ← 공통 유틸리티 모듈
│   ├── __init__.py
│   ├── data_loader.py              ← CSV 로드 & 전처리 (캐싱)
│   ├── ai_query.py                 ← OpenAI 호출 & 필터 실행
│   └── ui_helpers.py               ← 공통 CSS, 컴포넌트
│
├── data/                           ← 데이터 파일
│   └── 설비_고장률_정보.csv         ← 원본 CSV (여기에 위치)
│
├── assets/                         ← 이미지, 아이콘 등 (필요 시)
│
├── .streamlit/
│   └── config.toml                 ← 다크 테마, 포트 설정
│
├── .env.example                    ← 환경변수 예시 (복사 후 .env로 사용)
├── requirements.txt                ← Python 패키지 목록
└── README.md
```

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 입력
```

또는 앱 실행 후 사이드바에서 직접 API Key 입력 가능합니다.

### 3. 앱 실행

```bash
streamlit run Home.py
```

브라우저가 자동으로 열립니다: **http://localhost:8501**

---

## 📱 페이지 구성

| 페이지 | 설명 |
|---|---|
| **Home** | 시스템 소개 및 페이지 안내 |
| **자연어 조회** | AI가 질의를 분석하여 고장 데이터 필터링 |
| **현황 대시보드** | 설비별 TOP, 월별 트렌드, 카테고리 분포 차트 |

---

## 💬 자연어 질의 예시

| 유형 | 예시 |
|---|---|
| 키워드 검색 | "베어링 관련 고장 내역" |
| 기간 조건 | "이번 달 고장 목록" / "2024년 고장" |
| 설비+기간 | "크레인 최근 30일 고장" |
| 정렬 | "고장시간이 가장 긴 설비 TOP 10" |
| 작업반 | "기계정비1반이 처리한 고장" |
| 중요도 | "A등급 설비 고장만 보여줘" |

---

## 🔄 향후 DB 전환 (MySQL / MSSQL)

`utils/data_loader.py`의 `load_data()` 함수만 수정하면 됩니다:

```python
# MySQL 예시
import pymysql
import pandas as pd

@st.cache_data
def load_data():
    conn = pymysql.connect(
        host="서버IP", user="계정",
        password="비밀번호", db="DB명", charset="utf8"
    )
    df = pd.read_sql("SELECT * FROM 고장이력", conn)
    conn.close()
    return df
```

```python
# MSSQL 예시
import pyodbc
import pandas as pd

@st.cache_data
def load_data():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=서버IP;DATABASE=DB명;UID=계정;PWD=비밀번호"
    )
    df = pd.read_sql("SELECT * FROM 고장이력", conn)
    conn.close()
    return df
```

나머지 코드는 전혀 수정할 필요 없습니다.

---

## ⚙ 시스템 요구사항

- Python 3.9+
- 브라우저: Chrome / Edge 최신 버전
- 인터넷: OpenAI API 호출용 (AI 분석 기능만)
- 데이터: 사내 온프레미스 서버에서만 처리
