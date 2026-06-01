"""설비 고장정보 조회 시스템 — 메인 진입점"""

import streamlit as st

st.set_page_config(
    page_title="설비 고장정보 조회 시스템",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_helpers  import (inject_css, render_sidebar_stats, render_api_key_section,
                                render_query_conditions, get_query_conditions)
from utils.data_loader import load_data, get_summary_stats, apply_sidebar_filters

inject_css()
st.session_state["_page_id"] = "home"

try:
    df = load_data()
except Exception as e:
    df = None

with st.sidebar:
    if df is not None:
        # 1. 조회조건 (설비계층 + 기간 — 홈화면엔 결과건수 불필요하므로 숨김)
        cond = render_query_conditions(df)
        st.divider()

        # 2. 데이터 현황 (필터 적용 후)
        dff   = apply_sidebar_filters(df, cond)
        stats = get_summary_stats(dff)
        render_sidebar_stats(stats)
        st.divider()
    else:
        st.error("데이터 로드 실패")
        st.divider()

    # 3. API Key (맨 아래)
    render_api_key_section()

# ── 홈 화면 ──────────────────────────────────────────────
st.title("⚡ 설비 고장정보 조회 시스템")
st.markdown("AI 기반 자연어 질의로 고장 데이터를 손쉽게 조회하세요.")
st.divider()

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 🔍 자연어 조회")
    st.markdown("""
    AI가 자연어 질의를 분석하여 고장 데이터를 필터링합니다.
    - "이번 달 크레인 고장 건수는?"
    - "고장시간이 가장 긴 설비 TOP 10"
    - "베어링 관련 고장 내역"

    **왼쪽 메뉴 → 자연어 조회** 페이지로 이동하세요.
    """)
    if st.button("→ 자연어 조회 페이지로", type="primary", use_container_width=True):
        st.switch_page("pages/1_자연어_조회.py")

with col2:
    st.markdown("### 📊 현황 대시보드")
    st.markdown("""
    설비별 고장 현황을 차트와 통계로 한눈에 파악합니다.
    - 설비별 고장 TOP 10
    - 월별 고장 트렌드
    - 설비 카테고리별 분포

    **왼쪽 메뉴 → 현황 대시보드** 페이지로 이동하세요.
    """)
    if st.button("→ 현황 대시보드로", use_container_width=True):
        st.switch_page("pages/2_현황_대시보드.py")

st.divider()
st.caption("사내 인트라넷 전용 · 데이터는 서버에서만 처리됩니다.")
