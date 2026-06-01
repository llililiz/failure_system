"""
데이터 로딩 & 전처리 유틸리티
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CSV_PATH     = os.getenv("CSV_PATH", "data/설비_고장률_정보.csv")
CSV_ENCODING = os.getenv("CSV_ENCODING", "cp949")

# 원본 영문 컬럼 → 내부 처리용 영문 키
COLUMN_MAP = {
    "Planner Group":       "planner_group",
    "Equipment Category":  "equipment_category",
    "Object type":         "object_type",
    "Catalog Profile Code":"catalog_profile_code",
    "Catalog Profile":     "catalog_profile",
    "Equipment":           "equipment_id",
    "Tag Number":          "tag_number",
    "Equipment Desc.":     "equipment_desc",
    "Description":         "description",
    "Notification Date":   "notification_date",
    "Breakdown Duration":  "breakdown_duration",
    "Name of Work Center": "work_center",
    "Cost Center":         "cost_center",
    "Cost Center Desc":    "cost_center_desc",
    "Order":               "order_no",
    "Notification":        "notification_no",
    "ABC Indicator":       "abc_indicator",
    "Breakdown Indicator": "breakdown_indicator",
    "Object Part":         "object_part",
    "Problem or Damage":   "problem_or_damage",
    "Cause":               "cause",
    "Activity":            "activity",
}

# 화면 출력용 한글 컬럼명
DISPLAY_COLUMNS = {
    "notification_date":  "노티일",
    "equipment_category": "대분류",
    "object_type":        "중분류",
    "catalog_profile":    "소분류",
    "equipment_id":       "설비번호",
    "equipment_desc":     "설비명",
    "tag_number":         "태그번호",
    "description":        "상세내용",
    "breakdown_duration": "수리시간(h)",
    "work_center":        "정비조직",
    "cost_center_desc":   "사용부서",
    "abc_indicator":      "설비등급",
}

# 전체 한글 컬럼명 (상세 조회 등에서 참고용)
ALL_KO_COLUMNS = {
    "planner_group":      "플래너그룹",
    "equipment_category": "대분류",
    "object_type":        "중분류",
    "catalog_profile_code": "소분류코드",
    "catalog_profile":    "소분류",
    "equipment_id":       "설비번호",
    "tag_number":         "태그번호",
    "equipment_desc":     "설비명",
    "description":        "상세내용",
    "notification_date":  "노티일",
    "breakdown_duration": "수리시간",
    "work_center":        "정비조직",
    "cost_center":        "사용부서코드",
    "cost_center_desc":   "사용부서",
    "order_no":           "오더번호",
    "notification_no":    "노티번호",
    "abc_indicator":      "설비등급",
    "breakdown_indicator":"고장여부",
    "object_part":        "고장부위",
    "problem_or_damage":  "고장현상",
    "cause":              "고장원인",
    "activity":           "고장조치",
}


@st.cache_data(show_spinner="데이터 로딩 중...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding=CSV_ENCODING)
    df = df.rename(columns=COLUMN_MAP)

    df["notification_date"] = pd.to_datetime(df["notification_date"], errors="coerce")
    df["breakdown_duration"] = pd.to_numeric(df["breakdown_duration"], errors="coerce")

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].replace(r"^\s*$", pd.NA, regex=True)

    return df


def get_display_df(df: pd.DataFrame) -> pd.DataFrame:
    cols     = list(DISPLAY_COLUMNS.keys())
    existing = [c for c in cols if c in df.columns]
    out      = df[existing].copy()
    out["notification_date"] = out["notification_date"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns=DISPLAY_COLUMNS)
    return out


def get_summary_stats(df: pd.DataFrame) -> dict:
    """필터된 df 기준으로 통계 계산 (전체 df 또는 필터된 df 모두 받을 수 있음)"""
    return {
        "total":          len(df),
        "categories":     df["equipment_category"].nunique(),  # 대분류 수
        "obj_type_count": df["object_type"].nunique(),         # 중분류 수
        "equip_count":    df["equipment_desc"].nunique(),      # 설비 수
        "date_min":       df["notification_date"].min(),
        "date_max":       df["notification_date"].max(),
        "avg_dur":        round(df["breakdown_duration"].mean(), 1),
    }



def apply_sidebar_filters(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """조회조건 dict를 받아 df에 필터 적용 후 반환"""
    dff = df.copy()
    if cond.get("대분류"): dff = dff[dff["equipment_category"].isin(cond["대분류"])]
    if cond.get("중분류"): dff = dff[dff["object_type"].isin(cond["중분류"])]
    if cond.get("소분류"): dff = dff[dff["catalog_profile"].isin(cond["소분류"])]
    if cond.get("date_from"): dff = dff[dff["notification_date"] >= pd.Timestamp(cond["date_from"])]
    if cond.get("date_to"):   dff = dff[dff["notification_date"] <= pd.Timestamp(cond["date_to"])]
    return dff
