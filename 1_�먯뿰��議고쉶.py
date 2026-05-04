"""자연어 조회 페이지"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="자연어 조회 | 설비 고장정보",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_helpers  import (inject_css, render_sidebar_stats, render_ai_summary,
                                render_result_table, render_agg_table,
                                render_api_key_section, get_api_key,
                                render_query_conditions, get_query_conditions)
from utils.data_loader import load_data, get_summary_stats, apply_sidebar_filters
from utils.ai_query    import run_ai_query

inject_css()
st.session_state["_page_id"] = "query"

PLOTLY_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", margin=dict(l=0, r=0, t=44, b=0),
)

if "query_result" not in st.session_state:
    st.session_state.query_result = None

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ 데이터 로드 실패: {e}"); st.stop()

# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    # 1. 조회조건
    cond = render_query_conditions(df)
    st.divider()

    # 2. 데이터 현황 (필터 반영)
    dff   = apply_sidebar_filters(df, cond)
    stats = get_summary_stats(dff)
    render_sidebar_stats(stats)
    st.divider()

    # 3. API Key (맨 아래)
    render_api_key_section()

# ── 메인 ─────────────────────────────────────────────────
st.title("🔍 자연어 고장정보 조회")

with st.form(key="query_form", border=False):
    col_input, col_btn = st.columns([9, 1])
    with col_input:
        query_text = st.text_input(
            "질의 입력",
            placeholder=(
                "예) 용접기의 주요 고장원인과 조치결과는?  /  "
                "25년 크레인 고장건수  /  "
                "25년도 고장이 가장 많이난 용접기는?  /  "
                "크레인 주요 고장부위는?  /  "
                "베어링 관련 고장 내역  /  "
                "크레인의 평균 수리시간은?"
            ),
            label_visibility="collapsed",
            key="_query_input",
        )
    with col_btn:
        run_btn = st.form_submit_button("🔎 조회", type="primary", use_container_width=True)

# ── 조회 실행 ────────────────────────────────────────────
if run_btn:
    q = query_text.strip()
    if not q:
        st.warning("질의어를 입력하세요.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.warning("⚠️ OpenAI API Key가 없습니다.")
        else:
            cond = get_query_conditions()
            with st.spinner("AI가 질의를 분석하는 중..."):
                result = run_ai_query(
                    query=q, df=df, api_key=api_key,
                    limit=cond["limit"],
                    category_filter=cond["대분류"] or None,
                    object_type_filter=cond["중분류"] or None,
                    catalog_filter=cond["소분류"] or None,
                    date_from=cond["date_from"],
                    date_to=cond["date_to"],
                )
            st.session_state.query_result = result

# ── 결과 ─────────────────────────────────────────────────
result = st.session_state.query_result

if result is None:
    st.divider()
    st.info("💡 질의어를 입력하고 Enter 또는 조회 버튼을 누르세요.", icon="ℹ️")
    with st.expander("📖 사용 방법"):
        st.markdown("""
        | 질의 유형 | 예시 |
        |---|---|
        | 단순 건수 | "25년 용접기 고장건수" |
        | 단순 평균 | "크레인의 25년도 평균 수리시간은?" |
        | 집계·순위 | "25년도 고장이 가장 많이 난 용접기는?" |
        | 단일 분석 | "크레인의 주요 고장 원인은?" |
        | 복합 분석 | "용접기의 주요 고장원인과 조치결과는?" |
        | 목록 조회 | "베어링 관련 고장 내역" |
        """)

elif result.get("error"):
    st.error(f"❌ {result['error']}")
    with st.expander("🐛 생성된 코드"):
        st.code(result.get("filter_code",""), language="python")

else:
    result_type = result.get("result_type","list")
    if result.get("summary"):
        # \n\n을 <br><br>로 변환해 HTML에서 줄바꿈 표시
        render_ai_summary(result["summary"], result.get("query_intent",""))

    # ── COUNT ────────────────────────────────────────────
    if result_type == "count":
        st.metric("집계 결과", result.get("count_str",""))
        records_df = result.get("records", pd.DataFrame())
        if not records_df.empty:
            st.markdown("---")
            tab_list, tab_debug = st.tabs(["📋 해당 고장 목록", "🐛 생성된 코드"])
            with tab_list:
                render_result_table(records_df, result.get("total",0))
            with tab_debug:
                st.code(result.get("filter_code",""), language="python")

    # ── AGGREGATE ────────────────────────────────────────
    elif result_type == "aggregate":
        agg_df  = result.get("agg_result", pd.DataFrame())
        val_col = result.get("val_col","고장건수")
        if agg_df.empty:
            st.warning("집계 결과가 없습니다.")
        else:
            tab_chart, tab_table, tab_debug = st.tabs(["📊 차트","📋 집계 테이블","🐛 생성 코드"])
            with tab_chart:
                x_col = agg_df.columns[0]
                fmt   = "%{text:,.1f}" if agg_df[val_col].dtype==float else "%{text:,}"
                fig   = px.bar(agg_df, x=val_col, y=x_col, orientation="h",
                               color_discrete_sequence=["#4f8ef7"], text=val_col,
                               labels={x_col:"", val_col:val_col})
                fig.update_traces(texttemplate=fmt, textposition="outside")
                fig.update_layout(**PLOTLY_BASE, height=max(300,len(agg_df)*36+60))
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
            with tab_table: render_agg_table(agg_df)
            with tab_debug:
                st.code(result.get("filter_code",""), language="python")
                st.code(result.get("groupby_code",""), language="python")

    # ── FREQ (단일) ───────────────────────────────────────
    elif result_type == "freq":
        agg_df           = result.get("agg_result", pd.DataFrame())
        freq_label       = result.get("freq_label","")
        unclassified_info= result.get("unclassified_info")
        if agg_df.empty:
            st.warning("분석 결과가 없습니다.")
        else:
            tab_chart, tab_table, tab_debug = st.tabs(["📊 차트","📋 집계 테이블","🐛 생성 코드"])
            with tab_chart:
                x_col = agg_df.columns[0]
                fig   = px.bar(agg_df, x="빈도", y=x_col, orientation="h",
                               color_discrete_sequence=["#7b61ff"], text="빈도",
                               title=f"{freq_label} 빈도",
                               labels={x_col:freq_label or "", "빈도":"빈도(건)"})
                fig.update_traces(texttemplate="%{text:,}", textposition="outside")
                fig.update_layout(**PLOTLY_BASE, height=max(300,len(agg_df)*36+60))
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)

                # 미분류 정보 표시 (description 분석 시)
                if unclassified_info and unclassified_info["unclassified"] > 0:
                    ucl     = unclassified_info["unclassified"]
                    samples = unclassified_info["samples"]
                    with st.expander(f"📋 분류 못한 상세내역 {ucl:,}건 (샘플 100건)"):
                        for s in samples:
                            st.caption(f"• {s}")

            with tab_table: render_agg_table(agg_df)
            with tab_debug:
                st.code(result.get("filter_code",""), language="python")

    # ── MULTI_FREQ ────────────────────────────────────────
    elif result_type == "multi_freq":
        freq_results = result.get("freq_results", {})
        if not freq_results:
            st.warning("분석 결과가 없습니다.")
        else:
            tabs = st.tabs(["📊 차트","📋 테이블","🐛 코드"])
            with tabs[0]:
                items  = list(freq_results.items())
                colors = ["#7b61ff","#4f8ef7","#22c55e","#f59e0b"]
                for i in range(0, len(items), 2):
                    cols = st.columns(2)
                    for j, (label, kw_df) in enumerate(items[i:i+2]):
                        with cols[j]:
                            if kw_df.empty:
                                st.caption(f"{label}: 데이터 없음"); continue
                            x_col = kw_df.columns[0]
                            fig = px.bar(kw_df, x="빈도", y=x_col, orientation="h",
                                         color_discrete_sequence=[colors[(i+j)%4]],
                                         text="빈도", title=f"{label} 빈도",
                                         labels={x_col:label,"빈도":"빈도(건)"})
                            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
                            fig.update_layout(**PLOTLY_BASE, height=max(260,len(kw_df)*30+60))
                            fig.update_yaxes(categoryorder="total ascending")
                            st.plotly_chart(fig, use_container_width=True)
            with tabs[1]:
                for idx, (label, kw_df) in enumerate(freq_results.items()):
                    st.markdown(f"**{label}**")
                    render_agg_table(kw_df, key_suffix=f"{label}_{idx}") if not kw_df.empty else st.caption("데이터 없음")
            with tabs[2]:
                st.code(result.get("filter_code",""), language="python")

    # ── CROSS_FREQ: A별 B 교차 분석 ─────────────────────
    elif result_type == "cross_freq":
        cross_results = result.get("cross_results", {})
        group_label   = result.get("group_label", "")
        target_label  = result.get("target_label", "")

        if not cross_results:
            st.warning("교차 분석 결과가 없습니다.")
        else:
            tab_chart, tab_table, tab_debug = st.tabs(["📊 차트", "📋 테이블", "🐛 코드"])
            colors = ["#4f8ef7","#7b61ff","#22c55e","#f59e0b","#ef4444"]

            with tab_chart:
                items = list(cross_results.items())
                for i in range(0, len(items), 2):
                    cols = st.columns(2)
                    for j, (grp_val, info) in enumerate(items[i:i+2]):
                        with cols[j]:
                            kw_df = info["top"]
                            if kw_df.empty:
                                st.caption(f"{grp_val}: 데이터 없음")
                                continue
                            x_col = kw_df.columns[0]
                            fig = px.bar(
                                kw_df, x="빈도", y=x_col, orientation="h",
                                color_discrete_sequence=[colors[(i+j) % len(colors)]],
                                text="빈도",
                                title=f"{group_label}: {grp_val} ({info['count']:,}건)",
                                labels={x_col: target_label, "빈도": "빈도(건)"},
                            )
                            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
                            fig.update_layout(**PLOTLY_BASE, height=max(220, len(kw_df)*34+60))
                            fig.update_yaxes(categoryorder="total ascending")
                            st.plotly_chart(fig, use_container_width=True)

            with tab_table:
                for idx, (grp_val, info) in enumerate(cross_results.items()):
                    st.markdown(f"**{group_label}: {grp_val}** ({info['count']:,}건)")
                    if not info["top"].empty:
                        render_agg_table(info["top"], key_suffix=f"cross_{idx}")
                    else:
                        st.caption("데이터 없음")

            with tab_debug:
                st.code(result.get("filter_code",""), language="python")
                st.code(result.get("groupby_code",""), language="python")

    # ── GROUPED_MULTI_FREQ ───────────────────────────────
    elif result_type == "grouped_multi_freq":
        grouped_results = result.get("grouped_results", {})
        hints           = result.get("hints", [])

        if not grouped_results:
            st.warning("분석 결과가 없습니다.")
        else:
            tabs = st.tabs(["📊 차트", "📋 테이블", "🐛 코드"])
            colors = ["#4f8ef7","#7b61ff","#22c55e","#f59e0b","#ef4444",
                      "#06b6d4","#ec4899","#84cc16","#f97316","#a78bfa"]

            with tabs[0]:  # 차트
                for grp_idx, (grp_label, freq_dict) in enumerate(grouped_results.items()):
                    grp_cnt = list(freq_dict.values())[0]["count"] if freq_dict else 0
                    st.markdown(f"**{grp_label}** ({grp_cnt:,}건)")
                    freq_items = list(freq_dict.items())
                    cols = st.columns(len(freq_items)) if len(freq_items) > 1 else [st.container()]
                    for j, (flabel, finfo) in enumerate(freq_items):
                        with cols[j]:
                            kw_df = finfo["df"]
                            nc    = finfo["null"]
                            valid = grp_cnt - nc
                            if kw_df.empty:
                                st.caption(f"{flabel}: 데이터 없음")
                                continue
                            x_col = kw_df.columns[0]
                            null_note = f" (미입력 {nc:,}건 제외)" if nc > 0 else ""
                            fig = px.bar(
                                kw_df, x="빈도", y=x_col, orientation="h",
                                color_discrete_sequence=[colors[(grp_idx + j) % len(colors)]],
                                text="빈도",
                                title=f"{flabel}{null_note}",
                                labels={x_col: flabel, "빈도": "빈도(건)"},
                            )
                            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
                            fig.update_layout(**PLOTLY_BASE, height=max(200, len(kw_df)*32+60))
                            fig.update_yaxes(categoryorder="total ascending")
                            st.plotly_chart(fig, use_container_width=True)
                    st.divider()

            with tabs[1]:  # 테이블
                for grp_label, freq_dict in grouped_results.items():
                    grp_cnt = list(freq_dict.values())[0]["count"] if freq_dict else 0
                    st.markdown(f"**{grp_label}** ({grp_cnt:,}건)")
                    tcols = st.columns(len(freq_dict)) if len(freq_dict) > 1 else [st.container()]
                    for j, (flabel, finfo) in enumerate(freq_dict.items()):
                        with tcols[j]:
                            if not finfo["df"].empty:
                                render_agg_table(finfo["df"], key_suffix=f"{grp_label}_{flabel}_{j}")
                            else:
                                st.caption(f"{flabel}: 데이터 없음")
                    st.divider()

            with tabs[2]:  # 코드
                st.code(result.get("filter_code",""), language="python")
                st.code(result.get("groupby_code",""), language="python")

    # ── LIST ─────────────────────────────────────────────
    else:
        records_df = result.get("records", pd.DataFrame())
        total      = result.get("total",0)
        if total==0 or records_df.empty:
            st.warning("조건에 맞는 데이터가 없습니다.")
        else:
            tab_list, tab_agg, tab_debug = st.tabs(["📋 상세 목록","📊 간단 집계","🐛 생성 코드"])
            with tab_list:
                render_result_table(records_df, total)
            with tab_agg:
                c1, c2 = st.columns(2)
                with c1:
                    cat_cnt = records_df["equipment_category"].value_counts().reset_index()
                    cat_cnt.columns = ["대분류","건수"]
                    fig = px.bar(cat_cnt, x="건수", y="대분류", orientation="h",
                                 title="대분류별 건수", color_discrete_sequence=["#4f8ef7"], text="건수")
                    fig.update_traces(textposition="outside")
                    fig.update_layout(**PLOTLY_BASE, height=320)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    abc_cnt = records_df["abc_indicator"].value_counts().reset_index()
                    abc_cnt.columns = ["설비등급","건수"]
                    fig2 = px.pie(abc_cnt, names="설비등급", values="건수", title="설비등급별 분포",
                                  color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig2.update_layout(**PLOTLY_BASE, height=320)
                    st.plotly_chart(fig2, use_container_width=True)
                if "notification_date" in records_df.columns:
                    monthly = (records_df.assign(월=records_df["notification_date"].dt.to_period("M").astype(str))
                               .groupby("월").size().reset_index(name="건수"))
                    if len(monthly)>1:
                        fig3 = px.bar(monthly, x="월", y="건수", title="월별 고장 건수",
                                      color_discrete_sequence=["#7b61ff"])
                        fig3.update_layout(**PLOTLY_BASE, height=260)
                        st.plotly_chart(fig3, use_container_width=True)
            with tab_debug:
                st.code(result.get("filter_code",""), language="python")
