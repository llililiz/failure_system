"""
공통 UI 컴포넌트 & 스타일 헬퍼
"""

import io
import pandas as pd
import streamlit as st


# ── 공통 CSS ────────────────────────────────────────────
COMMON_CSS = """
<style>
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
div[data-testid="metric-container"] {
    background: #1e2538;
    border: 1px solid rgba(99,120,180,0.2);
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #94a3b8 !important;
    letter-spacing: 0.5px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    color: #4f8ef7 !important;
    font-weight: 700;
}
.ai-summary {
    background: rgba(79,142,247,0.06);
    border-left: 3px solid #4f8ef7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 14px;
    line-height: 1.8;
    color: #e2e8f0;
}
.ai-label {
    font-size: 10px;
    letter-spacing: 1.2px;
    color: #4f8ef7;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 6px;
}
.intent-text {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 6px;
}
.stButton > button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
    border-radius: 7px !important;
    border: 1px solid rgba(99,120,180,0.2) !important;
    background: #1e2538 !important;
    color: #94a3b8 !important;
    margin-bottom: 2px;
    transition: all .15s;
}
.stButton > button:hover {
    background: #252d40 !important;
    color: #e2e8f0 !important;
    border-color: rgba(99,120,180,0.4) !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid rgba(99,120,180,0.15);
    overflow: hidden;
}
.section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin: 16px 0 8px 0;
}
.badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 9px;
    border-radius: 20px;
    font-weight: 500;
    font-family: monospace;
}
.badge-blue  { background: rgba(79,142,247,.15); color: #4f8ef7; }
.badge-green { background: rgba(34,197,94,.15);  color: #22c55e; }
.badge-warn  { background: rgba(245,158,11,.15); color: #f59e0b; }
</style>
"""


def inject_css():
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


# ── Enter 키 → 조회 버튼 클릭 JS ────────────────────────
ENTER_KEY_JS = """
<script>
(function() {
    function attachListener() {
        var textarea = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
        if (!textarea) { setTimeout(attachListener, 300); return; }
        if (textarea._enterBound) return;
        textarea._enterBound = true;
        textarea.addEventListener('keydown', function(e) {
            // Shift+Enter → 줄바꿈 (기본 동작 유지)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                // data-testid="baseButton-primary" 인 첫 번째 버튼(조회)을 클릭
                var btn = window.parent.document.querySelector('button[data-testid="baseButton-primary"]');
                if (btn) btn.click();
            }
        });
    }
    attachListener();
})();
</script>
"""


def inject_enter_key_js():
    """textarea에서 Enter → 조회, Shift+Enter → 줄바꿈"""
    st.components.v1.html(ENTER_KEY_JS, height=0)


# ── API Key ──────────────────────────────────────────────
def get_api_key() -> str:
    """우선순위: secrets.toml → session_state(사이드바 수동 입력)"""
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except (KeyError, AttributeError, Exception):
        pass
    return st.session_state.get("api_key", "")


def render_api_key_section():
    """secrets.toml 키 있으면 '적용됨' 표시, 없으면 입력란"""
    try:
        secret_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, AttributeError, Exception):
        secret_key = ""

    if secret_key:
        st.success("🔑 API Key: secrets.toml 적용됨", icon="✅")
    else:
        with st.expander("🔑 OpenAI API Key 입력", expanded=not st.session_state.get("api_key")):
            key_input = st.text_input(
                "API Key",
                value=st.session_state.get("api_key", ""),
                type="password",
                placeholder="sk-...",
            )
            if key_input:
                st.session_state["api_key"] = key_input
                st.success("✓ 저장됨", icon="✅")


# ── 사이드바 메트릭 ─────────────────────────────────────
def render_sidebar_stats(stats: dict):
    st.markdown('<div class="section-header">📊 데이터 현황</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("총 고장건수", f"{stats['total']:,}")
    c2.metric("설비 중분류", stats.get("obj_type_count", "-"))
    c1.metric("평균 수리시간", f"{stats['avg_dur']}h")
    c2.metric("설비 수", f"{stats.get('equip_count', 0):,}")


# ── 예시 질의 버튼 ──────────────────────────────────────
EXAMPLE_QUERIES = [
    "이번 달 크레인 고장 건수",
    "고장시간이 가장 긴 설비 TOP 10",
    "베어링 관련 고장 내역",
    "호이스트 작동 불량 목록",
    "2024년 월별 고장 건수",
    "기계정비1반 처리 고장",
    "인버터 폴트 고장 현황",
    "최근 30일 A등급 설비 고장",
    "이번 달 고장이 가장 많은 설비종류",
    "올해 설비별 고장 횟수 집계",
]


def render_example_queries() -> str | None:
    st.markdown('<div class="section-header">💬 질의 예시</div>', unsafe_allow_html=True)
    for q in EXAMPLE_QUERIES:
        if st.button(f"  {q}", key=f"eq_{q}"):
            return q
    return None


# ── AI 요약 박스 ────────────────────────────────────────
def render_ai_summary(summary: str, intent: str = ""):
    # \n\n → <br><br>, \n → <br> 변환 (HTML 줄바꿈 처리)
    summary_html = summary.replace("\n\n", "<br><br>").replace("\n", "<br>")
    html = f"""
    <div class="ai-summary">
        <div class="ai-label">✦ AI 분석 요약</div>
        {"<div class='intent-text'>의도: " + intent + "</div>" if intent else ""}
        {summary_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── 결과 테이블 ─────────────────────────────────────────
def render_result_table(df: pd.DataFrame, total: int):
    from utils.data_loader import get_display_df

    display = get_display_df(df)

    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown(
            f'<span class="badge badge-blue">전체 {total:,}건 중 {len(df):,}건 표시</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.download_button(
            "⬇ CSV 다운로드",
            data=to_csv_bytes(display),
            file_name=f"고장조회_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        display,
        use_container_width=True,
        height=480,
        column_config={
            "수리시간(h)": st.column_config.NumberColumn(format="%.1f h"),
            "노티일":      st.column_config.TextColumn(width="small"),
            "대분류":      st.column_config.TextColumn(width="small"),
            "중분류":      st.column_config.TextColumn(width="medium"),
            "소분류":      st.column_config.TextColumn(width="medium"),
            "설비번호":    st.column_config.NumberColumn(format="%d"),
            "설비명":      st.column_config.TextColumn(width="large"),
            "설비등급":    st.column_config.TextColumn(width="small"),
            "태그번호":    st.column_config.TextColumn(width="small"),
        },
    )


# ── 집계 결과 테이블 ─────────────────────────────────────
def render_agg_table(agg_df: pd.DataFrame, key_suffix: str = ""):
    """집계 결과(groupby 결과)를 보기 좋게 출력"""
    import time
    unique_key = f"dl_agg_{key_suffix}_{int(time.time()*1000) % 99999}"
    col1, col2 = st.columns([6, 2])
    with col2:
        st.download_button(
            "⬇ CSV 다운로드",
            data=to_csv_bytes(agg_df),
            file_name=f"집계결과_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=unique_key,
        )
    st.dataframe(agg_df, use_container_width=True, height=420)


# ── 유틸 ────────────────────────────────────────────────
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return ("\ufeff" + df.to_csv(index=False)).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="고장조회결과")
    return buf.getvalue()


# ── 공통 설비 계층 필터 (전 페이지 session_state 공유) ──────
FILTER_KEY_L1 = "filter_대분류"
FILTER_KEY_L2 = "filter_중분류"
FILTER_KEY_L3 = "filter_소분류"


def render_equipment_filter(df: "pd.DataFrame") -> dict:
    """
    대분류 → 중분류 → 소분류 cascading 필터.
    선택값은 session_state에 저장되어 모든 페이지에서 공유됩니다.
    반환: {"대분류": [...], "중분류": [...], "소분류": [...]}
    """
    st.markdown('<div class="section-header">🏭 설비 필터</div>', unsafe_allow_html=True)

    # ── 대분류 ──
    all_l1 = sorted(df["equipment_category"].dropna().unique().tolist())
    sel_l1 = st.multiselect(
        "설비 대분류",
        options=all_l1,
        default=st.session_state.get(FILTER_KEY_L1, []),
        key=f"_eq_l1_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L1] = sel_l1

    # ── 중분류 (대분류 연동) ──
    df_l2 = df[df["equipment_category"].isin(sel_l1)] if sel_l1 else df
    all_l2 = sorted(df_l2["object_type"].dropna().unique().tolist())
    # 선택된 중분류 중 현재 목록에 없는 건 제거
    prev_l2 = [v for v in st.session_state.get(FILTER_KEY_L2, []) if v in all_l2]
    sel_l2 = st.multiselect(
        "설비 중분류",
        options=all_l2,
        default=prev_l2,
        key=f"_eq_l2_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L2] = sel_l2

    # ── 소분류 (중분류 연동) ──
    df_l3 = df_l2[df_l2["object_type"].isin(sel_l2)] if sel_l2 else df_l2
    all_l3 = sorted(df_l3["catalog_profile"].dropna().unique().tolist())
    prev_l3 = [v for v in st.session_state.get(FILTER_KEY_L3, []) if v in all_l3]
    sel_l3 = st.multiselect(
        "설비 소분류",
        options=all_l3,
        default=prev_l3,
        key=f"_eq_l3_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L3] = sel_l3

    return {"대분류": sel_l1, "중분류": sel_l2, "소분류": sel_l3}


def get_filter_category() -> list:
    """현재 선택된 대분류 목록 반환 (ai_query category_filter 인자용)"""
    return st.session_state.get(FILTER_KEY_L1, [])


# ── 통합 조회조건 필터 (설비계층 + 기간 + 결과건수) ───────
FILTER_KEY_DATE_FROM = "filter_date_from"
FILTER_KEY_DATE_TO   = "filter_date_to"
FILTER_KEY_LIMIT     = "filter_limit"


def render_query_conditions(df: "pd.DataFrame") -> dict:
    """
    '조회조건' 섹션: 설비 대/중/소분류 + 기간(from~to) + 결과건수
    모든 값은 session_state에 저장되어 페이지 간 공유됩니다.
    반환: {"대분류":[], "중분류":[], "소분류":[], "date_from":date, "date_to":date, "limit":int}
    """
    st.markdown('<div class="section-header">🔎 조회조건</div>', unsafe_allow_html=True)

    # ── 대분류 ──
    all_l1 = sorted(df["equipment_category"].dropna().unique().tolist())
    sel_l1 = st.multiselect(
        "설비 대분류",
        options=all_l1,
        default=st.session_state.get(FILTER_KEY_L1, []),
        key=f"_eq_l1_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L1] = sel_l1

    # ── 중분류 ──
    df_l2  = df[df["equipment_category"].isin(sel_l1)] if sel_l1 else df
    all_l2 = sorted(df_l2["object_type"].dropna().unique().tolist())
    prev_l2 = [v for v in st.session_state.get(FILTER_KEY_L2, []) if v in all_l2]
    sel_l2 = st.multiselect(
        "설비 중분류",
        options=all_l2,
        default=prev_l2,
        key=f"_eq_l2_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L2] = sel_l2

    # ── 소분류 ──
    df_l3  = df_l2[df_l2["object_type"].isin(sel_l2)] if sel_l2 else df_l2
    all_l3 = sorted(df_l3["catalog_profile"].dropna().unique().tolist())
    prev_l3 = [v for v in st.session_state.get(FILTER_KEY_L3, []) if v in all_l3]
    sel_l3 = st.multiselect(
        "설비 소분류",
        options=all_l3,
        default=prev_l3,
        key=f"_eq_l3_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_L3] = sel_l3

    # ── 조회기간 (한 행에 from~to) ──
    data_min = df["notification_date"].min().date()
    data_max = df["notification_date"].max().date()
    prev_from = st.session_state.get(FILTER_KEY_DATE_FROM, data_min)
    prev_to   = st.session_state.get(FILTER_KEY_DATE_TO,   data_max)
    st.markdown('<p style="font-size:14px;margin:8px 0 4px 0;color:#e2e8f0">조회기간</p>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        date_from = st.date_input(
            "시작일", value=prev_from,
            min_value=data_min, max_value=data_max,
            format="YYYY-MM-DD",
            key=f"_date_from_{st.session_state.get('_page_id','home')}",
            label_visibility="collapsed",
        )
    with dc2:
        date_to = st.date_input(
            "종료일", value=prev_to,
            min_value=data_min, max_value=data_max,
            format="YYYY-MM-DD",
            key=f"_date_to_{st.session_state.get('_page_id','home')}",
            label_visibility="collapsed",
        )

    st.session_state[FILTER_KEY_DATE_FROM] = date_from
    st.session_state[FILTER_KEY_DATE_TO]   = date_to

    # ── 결과 건수 ──
    prev_limit = st.session_state.get(FILTER_KEY_LIMIT, 100)
    limit = st.selectbox(
        "결과 건수",
        options=[50, 100, 200, 500],
        index=[50, 100, 200, 500].index(prev_limit) if prev_limit in [50,100,200,500] else 1,
        key=f"_limit_{st.session_state.get('_page_id','home')}",
    )
    st.session_state[FILTER_KEY_LIMIT] = limit

    return {
        "대분류":    sel_l1,
        "중분류":    sel_l2,
        "소분류":    sel_l3,
        "date_from": date_from,
        "date_to":   date_to,
        "limit":     limit,
    }


def get_query_conditions() -> dict:
    """현재 선택된 조회조건을 session_state에서 읽기"""
    import datetime as dt
    return {
        "대분류":    st.session_state.get(FILTER_KEY_L1, []),
        "중분류":    st.session_state.get(FILTER_KEY_L2, []),
        "소분류":    st.session_state.get(FILTER_KEY_L3, []),
        "date_from": st.session_state.get(FILTER_KEY_DATE_FROM, None),
        "date_to":   st.session_state.get(FILTER_KEY_DATE_TO,   None),
        "limit":     st.session_state.get(FILTER_KEY_LIMIT, 100),
    }
