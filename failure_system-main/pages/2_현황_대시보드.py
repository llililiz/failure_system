"""현황 대시보드 페이지"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="현황 대시보드 | 설비 고장정보",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_helpers  import inject_css, render_sidebar_stats, render_api_key_section, render_query_conditions, get_query_conditions
from utils.data_loader import load_data, get_summary_stats, apply_sidebar_filters, get_display_df
from utils.ui_helpers  import to_csv_bytes

inject_css()
st.session_state["_page_id"] = "dashboard"

CHART_COLORS = ["#4f8ef7","#7b61ff","#22c55e","#f59e0b","#ef4444",
                "#06b6d4","#ec4899","#84cc16","#f97316","#a78bfa"]
PLOTLY_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", margin=dict(l=0, r=0, t=44, b=0),
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}"); st.stop()

# ── 사이드바 ────────────────────────────────────────────
with st.sidebar:
    # 1. 조회조건 (공통 필터)
    cond  = render_query_conditions(df)
    st.divider()

    # 2. TOP N 슬라이더
    top_n = st.slider("TOP N 설비", 5, 30, 10)
    st.divider()

    # 3. 데이터 현황 (필터 반영)
    dff_sidebar = apply_sidebar_filters(df, cond)
    stats_sb    = get_summary_stats(dff_sidebar)
    render_sidebar_stats(stats_sb)
    st.divider()

    # 4. API Key (맨 아래)
    render_api_key_section()

# ── 필터 적용 ────────────────────────────────────────────
cond = get_query_conditions()
dff  = apply_sidebar_filters(df, cond)

# ── 메인 ────────────────────────────────────────────────
st.title("📊 고장 현황 대시보드")
st.caption(f"조회 건수: {len(dff):,}건 / 전체 {len(df):,}건")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("총 고장건수",     f"{len(dff):,}")
m2.metric("평균 수리시간",   f"{dff['breakdown_duration'].mean():.1f}h")
m3.metric("최대 수리시간",   f"{dff['breakdown_duration'].max():.0f}h")
m4.metric("A등급 설비 고장", f"{(dff['abc_indicator']=='A').sum():,}")
m5.metric("관련 설비 수",    f"{dff['equipment_desc'].nunique():,}")

st.divider()

row1_l, row1_r = st.columns([1, 1], gap="large")
with row1_l:
    st.subheader(f"🏆 고장 TOP {top_n} 설비")
    top_equip = (dff.groupby(["equipment_desc", "equipment_category"])
                 .agg(건수=("description","count"), 총수리시간=("breakdown_duration","sum"))
                 .reset_index().sort_values("건수", ascending=False).head(top_n))
    fig = px.bar(top_equip.sort_values("건수"), x="건수", y="equipment_desc",
                 orientation="h", color="equipment_category",
                 color_discrete_sequence=CHART_COLORS,
                 hover_data={"총수리시간":":.1f"},
                 labels={"equipment_desc":"설비명","equipment_category":"대분류"})
    fig.update_layout(**PLOTLY_BASE, height=380, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

with row1_r:
    st.subheader("📈 월별 고장 추이")
    monthly = (dff.assign(월=dff["notification_date"].dt.to_period("M").astype(str))
               .groupby(["월","equipment_category"]).size().reset_index(name="건수"))
    fig2 = px.bar(monthly, x="월", y="건수", color="equipment_category",
                  color_discrete_sequence=CHART_COLORS,
                  labels={"equipment_category":"대분류"})
    fig2.update_layout(**PLOTLY_BASE, height=380, legend=dict(orientation="h", y=-0.15))
    fig2.update_xaxes(tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

row2_l, row2_m, row2_r = st.columns(3, gap="large")
with row2_l:
    st.subheader("🏭 대분류별 고장")
    cat_cnt = dff["equipment_category"].value_counts().reset_index()
    cat_cnt.columns = ["대분류","건수"]
    fig3 = px.pie(cat_cnt, names="대분류", values="건수",
                  color_discrete_sequence=CHART_COLORS, hole=0.45)
    fig3.update_layout(**PLOTLY_BASE, height=300)
    fig3.update_traces(textposition="outside", textfont_size=11)
    st.plotly_chart(fig3, use_container_width=True)

with row2_m:
    st.subheader("⭐ 설비등급별 고장")
    abc_cnt = dff["abc_indicator"].value_counts().reset_index()
    abc_cnt.columns = ["설비등급","건수"]
    color_map = {"A":"#ef4444","B":"#f59e0b","C":"#22c55e","D":"#94a3b8"}
    fig4 = px.bar(abc_cnt, x="설비등급", y="건수", color="설비등급",
                  color_discrete_map=color_map, text="건수")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(**PLOTLY_BASE, height=300, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

with row2_r:
    st.subheader("⏱ 수리시간 분포")
    dur = dff["breakdown_duration"].dropna()
    fig5 = px.histogram(dur, nbins=30,
                        labels={"value":"수리시간(h)","count":"건수"},
                        color_discrete_sequence=["#7b61ff"])
    fig5.update_layout(**PLOTLY_BASE, height=300)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

st.subheader("🔧 정비조직별 고장 처리 현황")
wc = (dff.groupby("work_center")
      .agg(건수=("description","count"), 평균수리시간=("breakdown_duration","mean"))
      .reset_index().sort_values("건수", ascending=False).head(15))
wc["평균수리시간"] = wc["평균수리시간"].round(1)

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=wc["work_center"], y=wc["건수"],
                      name="고장 건수", marker_color="#4f8ef7", yaxis="y"))
fig6.add_trace(go.Scatter(x=wc["work_center"], y=wc["평균수리시간"],
                          name="평균 수리시간(h)", mode="lines+markers",
                          marker=dict(color="#f59e0b", size=7),
                          line=dict(color="#f59e0b"), yaxis="y2"))
fig6.update_layout(
    **PLOTLY_BASE, height=340,
    yaxis=dict(title="고장 건수", gridcolor="rgba(99,120,180,0.1)"),
    yaxis2=dict(title="평균 수리시간(h)", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.1),
    xaxis=dict(tickangle=-30),
)
st.plotly_chart(fig6, use_container_width=True)

st.divider()
with st.expander("📋 전체 데이터 보기"):
    display_df = get_display_df(dff)
    st.dataframe(display_df, use_container_width=True, height=400)
    st.download_button(
        "⬇ 전체 CSV 다운로드",
        data=to_csv_bytes(display_df),
        file_name=f"고장현황_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
