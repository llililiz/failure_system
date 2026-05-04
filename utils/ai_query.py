"""
자연어 질의 → OpenAI → pandas 실행

result_type:
  "count"      : 단순 건수/합계/평균
  "aggregate"  : 그룹별 집계 TOP N
  "list"       : raw 레코드 목록
  "freq"       : 고장부위/현상/원인/조치 빈도 (복수 가능)
  "multi_freq" : 복수 freq 동시 분석
"""

import os
import json
import re
import pandas as pd
from collections import Counter
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from utils.fuzzy_match import extract_equipment_filter, apply_equipment_filter

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

FREQ_COL_MAP = {
    "고장부위": "object_part",
    "고장현상": "problem_or_damage",
    "고장원인": "cause",
    "고장조치": "activity",
    "상세내용": "description",
}

# cross_freq: 그룹(group_col)별 대상(target_col)의 빈도 교차 분석
# 예: "고장현상별 조치결과" → group_col=problem_or_damage, target_col=activity


def build_system_prompt(df: pd.DataFrame) -> str:
    cats          = df["equipment_category"].dropna().unique().tolist()
    obj_types     = df["object_type"].dropna().unique().tolist()[:20]
    work_centers  = df["work_center"].dropna().unique().tolist()[:15]
    sample_desc   = df["description"].dropna().head(8).tolist()
    today_str     = datetime.now().strftime("%Y-%m-%d")
    this_year     = datetime.now().year
    this_month    = datetime.now().month
    sample_cause  = df["cause"].dropna().value_counts().head(6).index.tolist()
    sample_prob   = df["problem_or_damage"].dropna().value_counts().head(6).index.tolist()
    sample_part   = df["object_part"].dropna().value_counts().head(6).index.tolist()
    sample_act    = df["activity"].dropna().value_counts().head(6).index.tolist()

    return (
        "당신은 설비 고장 데이터 분석 전문가입니다.\n"
        "사용자 질의를 분석하여 JSON만 반환합니다. 마크다운 코드블록이나 설명 텍스트 없이 JSON만 출력하세요.\n\n"

        "【데이터 컬럼】\n"
        f"- equipment_category : 설비 대분류 (값: {cats})\n"
        f"- object_type        : 설비 중분류 (예: {obj_types[:6]})\n"
        "- equipment_desc     : 설비명\n"
        "- description        : 상세내용 (자유 텍스트)\n"
        "- notification_date  : 노티일 (datetime64)\n"
        "- breakdown_duration : 수리시간 (float, 시간)\n"
        f"- work_center        : 정비조직\n"
        "- abc_indicator      : 설비등급 (A~D)\n"
        f"- object_part        : 고장부위 (예: {sample_part})\n"
        f"- problem_or_damage  : 고장현상 (예: {sample_prob})\n"
        f"- cause              : 고장원인 (예: {sample_cause})\n"
        f"- activity           : 고장조치 (예: {sample_act})\n\n"

        f"【오늘 날짜】 {today_str} (이번 연도={this_year}, 이번 달={this_month})\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "【반환 JSON 스키마】\n"
        "{\n"
        '  "result_type" : "count" | "aggregate" | "list" | "freq" | "multi_freq" | "cross_freq" | "grouped_multi_freq",\n'
        '  "filter_expr" : "반드시 df[조건] 형태의 완전한 pandas 식",\n'
        '  "agg_col"     : "집계 컬럼 (count/aggregate 전용)",\n'
        '  "agg_func"    : "count | sum | mean",\n'
        '  "groupby_col" : "groupby 기준 컬럼 (aggregate 전용)",\n'
        '  "top_n"       : 10,\n'
        '  "freq_col"    : "고장원인|고장현상|고장부위|고장조치|상세내용 (freq 단일 전용)",\n'
        '  "freq_cols"   : ["고장원인","고장조치"] (multi_freq 전용, 2개 이상),\n'
        '  "groupby_col" : "그룹 기준 컬럼 — 반드시 아래 규칙 준수",\n'
        '  "summary"     : "한국어 요약. 플레이스홀더: PLACEHOLDER_COUNT, PLACEHOLDER_VAL, PLACEHOLDER_TOP_NAME, PLACEHOLDER_TOP_VAL",\n'
        '  "query_intent": "질의 의도 한 줄"\n'
        "}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "【result_type 판단 — 가장 중요, 반드시 엄수】\n\n"
        '"count"      — 단일 숫자 하나. "몇 건", "건수는", "합계", "평균 수리시간은?"\n'
        '  ⚠️ 단순 평균/합계 → count. "가장 긴" 없으면 aggregate 아님\n\n'
        '"aggregate"  — 그룹별 TOP N 순위. "가장 많은", "TOP N", "어느 설비", "가장 긴"\n\n'
        '"freq"       — 1가지 분석 컬럼의 빈도만 볼 때\n'
        '  예) "크레인의 주요 고장 원인은?" → freq, freq_col="고장원인"\n\n'
        '"multi_freq" — 2가지 이상 컬럼을 각각 독립적으로 분석 (A와 B가 서로 관계없이 각각 빈도)\n'
        '  예) "용접기의 주요 고장원인과 조치결과는?" → multi_freq, freq_cols=["고장원인","고장조치"]\n\n'
        '"cross_freq" — A별 B 교차분석. A유형마다 B가 어떻게 다른지 볼 때\n'
        '  핵심 판단 키워드: "A별 B", "A에 따른 B", "A일 때 B", "A 유형별 B", "A별 주요 B"\n'
        '  예) "크레인의 주요 고장증상별 주요 조치결과는?" → cross_freq, freq_cols=["고장현상","고장조치"]\n'
        '  예) "고장부위별 주요 고장원인은?" → cross_freq, freq_cols=["고장부위","고장원인"]\n'
        '  freq_cols: [그룹기준, 분석대상] 반드시 2개만\n\n'
        '"list"       — 레코드 목록. "보여줘", "내역", "목록"\n\n'
        '"grouped_multi_freq" — 설비분류(대/중/소분류) 그룹별로 복수 freq 분석\n'
        '  판단 키워드: "대분류별","중분류별","소분류별" + freq 항목 1개 이상\n'
        '  예) "소분류 기준으로 주로 발생하는 고장현상과 그 원인" → grouped_multi_freq\n'
        '  예) "중분류별 주요 고장원인과 조치" → grouped_multi_freq\n'
        '  groupby_col 규칙:\n'
        '    "대분류별", "대분류 기준" → groupby_col = "equipment_category"\n'
        '    "중분류별", "중분류 기준" → groupby_col = "object_type"\n'
        '    "소분류별", "소분류 기준" → groupby_col = "catalog_profile"  ← 기본값\n\n'
        '  freq_cols: ["고장현상","고장원인"] 등 분석 항목 1~2개\n\n'
        "【freq_col / freq_cols 값 — 반드시 아래 중 하나】\n"
        '"고장원인" → cause 컬럼 (구조화 코드값: 노후/과부하/기타 등)\n'
        '"고장현상" → problem_or_damage 컬럼 (이상/정지/파손 등)\n'
        '"고장부위" → object_part 컬럼 (권상부/전장부 등)\n'
        '"고장조치" → activity 컬럼 (교체/복귀/점검 등)\n'
        '"상세내용" → description 컬럼 (자유텍스트 키워드 분석)\n'
        '  ⚠️ "상세내역", "상세내용", "설명", "내용 상", "기록 상" 등 → 반드시 freq_col="상세내용"\n'
        '  ⚠️ "고장원인 컬럼"이 아닌 "상세내역 글에서 원인을 파악"하는 질의 → freq_col="상세내용"\n\n'
        "【filter_expr 규칙 — 매우 중요】\n"
        "⚠️ filter_expr은 반드시 df[...] 형태. 조건식만 반환 절대 금지.\n"
        "\n"
        "【이미 적용된 사이드바 필터】\n"
        "설비 대분류(equipment_category), 기간(notification_date from~to)이 이미 적용됨.\n"
        "따라서 filter_expr에 equipment_category나 기간을 중복으로 추가하지 말 것.\n"
        "\n"
        "【RULE 1 — 기간 미언급 시 반드시 filter_expr = 'df'】\n"
        "질의에 기간(연도/월/날짜)이 명시적으로 없으면 → filter_expr = 'df'\n"
        "절대로 이번 달, 이번 해, 올해, 최근으로 임의 제한하지 말 것.\n"
        "기간 미언급 예: '용접기 주요 고장원인', '크레인 고장 목록', '평균 수리시간은?'\n"
        "→ 모두 filter_expr = 'df'\n"
        "\n"
        "【RULE 2 — 기간이 명시된 경우에만 기간 조건 추가】\n"
        "기간 명시 예: '25년 용접기', '이번 달 크레인', '최근 30일'\n"
        f"- 이번 달: df[(df['notification_date'].dt.year=={this_year})&(df['notification_date'].dt.month=={this_month})]\n"
        f"- 25년/2025년/올해: df[df['notification_date'].dt.year==2025]\n"
        f"- 24년/2024년: df[df['notification_date'].dt.year==2024]\n"
        "- N년(2자리): df[df['notification_date'].dt.year==(2000+N)]\n"
        f"- 최근 N일: df[df['notification_date']>=pd.Timestamp('{today_str}')-pd.Timedelta(days=N)]\n"
        "\n"
        "【RULE 3 — 텍스트 검색】\n"
        "- 단독: df[df['description'].str.contains('키워드',na=False,case=False)]\n"
        "- 기간+텍스트: df[(df['notification_date'].dt.year==2025)&(df['description'].str.contains('베어링',na=False,case=False))]\n"
        "\n"
        "【summary 플레이스홀더】\n"
        "- count+count    : 'PLACEHOLDER_COUNT건입니다.'\n"
        "- count+mean/sum : 'PLACEHOLDER_VAL시간입니다.'\n"
        "- aggregate      : 'PLACEHOLDER_TOP_NAME으로 PLACEHOLDER_TOP_VAL건입니다.'\n"
        "- freq/multi_freq: 'PLACEHOLDER_COUNT건 분석 결과입니다.' (서버가 상세 내용 자동 생성)\n"
        "- list           : 'PLACEHOLDER_COUNT건입니다.'\n"
    )


# ── 빈도 분석 헬퍼 ──────────────────────────────────────
def _resolve_col(hint: str, df: pd.DataFrame) -> str:
    mapped = FREQ_COL_MAP.get(hint, hint)
    return mapped if mapped in df.columns else "cause"

def _value_counts_df(series: pd.Series, top_n: int, label: str) -> pd.DataFrame:
    counts = (series.dropna().astype(str).str.strip()
              .replace("", pd.NA).dropna()
              .value_counts().head(top_n).reset_index())
    counts.columns = [label, "빈도"]
    return counts

def _keyword_df(series: pd.Series, top_n: int) -> pd.DataFrame:
    STOPWORDS = {
        "발생","작동","점검","보수","수리","교체","정비","실시","확인","조치","완료","처리",
        "요망","요청","불능","긴급","돌발","운전","운행","사용","작업","설치","고정",
        "설비","기계","장비","현장","부품","시스템","인한","그리고","이후","하여",
        "때문","인해","위해","관련","중","상","하","및","등","를","을","이","가",
        "의","에","도","로","와","과","TON","EOC","LLC","SGC","LMC","SGH",
    }
    pattern = re.compile(r'[가-힣]{2,}|[A-Z][A-Z0-9]{2,}')
    counter: Counter = Counter()
    for text in series.dropna():
        tokens = [t for t in pattern.findall(str(text))
                  if t not in STOPWORDS and not (t.isupper() and len(t)<=2)]
        counter.update(tokens)
    df_kw = pd.DataFrame(counter.most_common(top_n), columns=["항목", "빈도"])
    return df_kw

def _analyze_description_with_ai(series: pd.Series, client, model: str,
                                   aspect: str = "고장원인", top_n: int = 20) -> dict:
    """
    description 자유텍스트에서 aspect별 핵심 키워드를 배타적(exclusive) 집계.
    - 긴 키워드 우선 처리 → 한 번 분류된 건은 다른 키워드로 재분류 안 함 (중복 제거)
    - 분류 못한 건수와 샘플도 반환

    반환: {
        "df": DataFrame(항목, 빈도),
        "classified": int,   # 분류된 건수
        "unclassified": int, # 미분류 건수
        "unclassified_samples": [str, ...],  # 미분류 샘플 10건
    }
    """
    from collections import Counter

    texts = series.dropna().reset_index(drop=True)
    if texts.empty:
        return {"df": pd.DataFrame(columns=["항목","빈도"]),
                "classified": 0, "unclassified": 0, "unclassified_samples": []}

    # ── aspect별 정규식 패턴 (label, regex) 목록 — 위에서부터 배타적 적용 ──
    ASPECT_PATTERNS = {
        "고장현상": [
            # 복합 현상 (구체적인 것 우선)
            (r"작동\s*불량|동작\s*불량|작동불랴",     "작동불량"),
            (r"작동\s*이상|동작\s*이상",              "작동이상"),
            (r"작동\s*불능|동작\s*불능|미동작|무동작", "작동불능"),
            (r"작동\s*불가|동작\s*불가|작동\s*안됨|동작\s*안됨|작동\s*못함", "작동불능"),
            (r"선회\s*불능|선회\s*불가|선회\s*안됨",  "작동불능"),
            (r"주행\s*불능|주행\s*불가|주행\s*안됨",  "주행불가"),
            (r"횡행\s*불능|횡행\s*불가|횡행\s*안됨",  "작동불능"),
            (r"권상\s*불능|권상\s*불가|권상\s*안됨",  "작동불능"),
            (r"미작동|무작동",                        "미작동"),
            (r"속도\s*불량|속도가\s*안남|속도\s*저하", "속도불량"),
            # 단순 현상
            (r"정지",                                 "정지"),
            (r"파손|절단|파단|탈락",                  "파손/탈락"),
            (r"소음|진동|떨림",                       "소음/진동"),
            (r"마모|마멸",                            "마모"),
            (r"쇼트|SHORT|short",                    "단선/단락"),
            (r"단선|단락",                            "단선/단락"),
            (r"누설|누유|누수",                       "누설"),
            (r"발열|과열",                            "발열"),
            # 영문/코드
            (r"FAULT|fault|폴트|폴스",                "FAULT"),
            (r"TRIP|trip",                            "TRIP"),
            (r"[Ee][Rr][Rr][Oo][Rr]|에러|애러",      "에러"),
            (r"overspeed|OVERSPEED",                  "과속"),
            # 포괄
            # 전원/제어 불량
            (r"전원\s*안들어|전원\s*불량|전원\s*이상|콘트롤\s*(온|ON)\s*(안됨|불능)|CONTROL\s*ON\s*안됨", "전원불량"),
            # 소손/탄화
            (r"소손|탄화|소실",                        "소손"),
            # 주행 관련 복합
            (r"주행\s*작동\s*이상|주행\s*작동\s*불량|주행작동이사",  "주행불량"),
            # Fault/Trip 알람 (작동 중 발생)
            (r"[Ff][Aa][Uu][Ll][Tt]\s*\d+\s*발생|[Ff]ault\s*발생|작동\s*시\s*[Ff]ault", "FAULT알람"),
            (r"Motor\s*[Tt]rip|모터\s*트립|[Mm]otor\s*Trip", "모터트립"),
            # 포괄
            (r"이상|불량",                            "이상/불량"),
        ],
        "고장원인": [
            (r"접촉\s*불량",                          "접촉불량"),
            (r"절연\s*불량",                          "절연불량"),
            (r"배선\s*불량",                          "배선불량"),
            (r"정비\s*불량|관리\s*불량",              "정비/관리불량"),
            (r"운전자\s*불량|사용자\s*부주의|부주의",  "사용자 부주의"),
            (r"과전류|누전",                          "전기적 원인"),
            (r"과부하",                               "과부하"),
            (r"마모|마멸",                            "마모"),
            (r"노후|열화|부식",                       "노후/열화"),
            (r"단선",                                 "단선"),
            (r"결함|불량",                            "결함/불량"),
            (r"충격",                                 "충격"),
            (r"오염|이물질",                          "오염"),
        ],
        "고장부위": [
            (r"리미트\s*스위치|LIMIT\s*S/W|LIMIT SW",  "리미트스위치"),
            (r"전자\s*접촉기|CONTACTOR",              "전자접촉기"),
            (r"유압\s*호스",                          "유압호스"),
            (r"유압\s*실린더",                        "유압실린더"),
            (r"BEARING|베어링",                       "베어링"),
            (r"인버터|INVERTER",                      "인버터"),
            (r"브레이크|BRAKE",                       "브레이크"),
            (r"모터|MOTOR",                           "모터"),
            (r"기어|GEAR",                            "기어"),
            (r"케이블|CABLE",                         "케이블"),
            (r"센서|SENSOR",                          "센서"),
            (r"밸브|VALVE",                           "밸브"),
            (r"펌프|PUMP",                            "펌프"),
            (r"드라이브|DRIVE",                       "드라이브"),
            (r"릴레이|RELAY",                         "릴레이"),
            (r"리모콘|REMOTE",                        "리모콘"),
            (r"호이스트|HOIST",                       "호이스트"),
            (r"유압",                                  "유압장치"),
        ],
        "고장조치": [
            (r"분해\s*조립|재조립",                   "분해조립"),
            (r"교체|교환",                            "교체"),
            (r"보수",                                  "보수"),
            (r"점검",                                  "점검"),
            (r"조정|조절",                            "조정"),
            (r"청소|세척",                            "청소"),
            (r"복귀|복원",                            "복귀"),
            (r"결선",                                  "결선"),
            (r"수정",                                  "수정"),
            (r"윤활|주유",                            "윤활"),
        ],
        "종합": [
            (r"작동\s*불량|동작\s*불량",              "작동불량"),
            (r"작동\s*이상",                          "작동이상"),
            (r"작동\s*불능|동작\s*불능|선회\s*불능",  "작동불능"),
            (r"미작동",                               "미작동"),
            (r"이상|불량",                            "이상/불량"),
            (r"정지|파손|마모",                       "기타현상"),
            (r"FAULT|fault|TRIP|trip",                "코드에러"),
            (r"교체|보수|점검",                       "조치"),
        ],
    }

    patterns = ASPECT_PATTERNS.get(aspect, ASPECT_PATTERNS["종합"])

    classified_mask = pd.Series([False] * len(texts), index=texts.index)
    result: Counter = Counter()

    for pat, label in patterns:
        unclassified = texts[~classified_mask]
        if unclassified.empty:
            break
        mask = unclassified.str.contains(pat, na=False, regex=True)
        cnt  = int(mask.sum())
        if cnt > 0:
            result[label] = result.get(label, 0) + cnt
            classified_mask = classified_mask | mask.reindex(texts.index, fill_value=False)

    total        = len(texts)
    classified   = int(classified_mask.sum())
    unclassified = total - classified
    samples      = texts[~classified_mask].head(100).tolist()

    if not result:
        return {"df": pd.DataFrame(columns=["항목","빈도"]),
                "classified": 0, "unclassified": total, "unclassified_samples": samples}

    df_result = pd.DataFrame(result.most_common(top_n), columns=["항목", "빈도"])
    return {
        "df": df_result,
        "classified": classified,
        "unclassified": unclassified,
        "unclassified_samples": samples,
    }

def _freq_summary(kw_df: pd.DataFrame, total: int, col_label: str,
                  show_total: bool = True, null_count: int = 0,
                  filter_note: str = "") -> str:
    """freq 요약 문장 생성
    total: 필터된 전체 건수
    null_count: 해당 컬럼 값이 없는 건수
    filter_note: 적용된 설비 필터 설명 (예: "크레인")
    """
    valid = total - null_count
    if kw_df.empty:
        scope = f"[{filter_note}] " if filter_note else ""
        return f"{scope}총 {total:,}건 중 {col_label} 데이터가 없습니다."
    top5    = " · ".join(kw_df.iloc[:, 0].head(5).tolist())
    top1_nm = kw_df.iloc[0, 0]
    top1_v  = int(kw_df.iloc[0]["빈도"])
    pct     = round(top1_v / valid * 100, 1) if valid > 0 else 0

    lines = []
    if show_total:
        scope     = f"[{filter_note}] " if filter_note else ""
        null_note = f", 미입력 {null_count:,}건 제외" if null_count > 0 else ""
        lines.append(f"{scope}총 {total:,}건 중 {col_label} 입력 {valid:,}건{null_note} 분석")
    lines.append(f"주요 {col_label}: [{top5}] 순. 1위 '{top1_nm}' {top1_v:,}건({pct}%)")
    return "\n\n".join(lines)


def run_ai_query(
    query: str,
    df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_MODEL,
    limit: int = 200,
    category_filter: list = None,
    object_type_filter: list = None,
    catalog_filter: list = None,
    date_from=None,
    date_to=None,
) -> dict:
    if not api_key:
        return {"error": "OpenAI API Key가 없습니다."}

    client  = OpenAI(api_key=api_key)
    base_df = df.copy()

    # ── 1. 사이드바 기간 필터 선적용 ──
    if date_from:
        base_df = base_df[base_df["notification_date"] >= pd.Timestamp(date_from)]
    if date_to:
        base_df = base_df[base_df["notification_date"] <= pd.Timestamp(date_to)]


    # ── 1-1. 사이드바 대/중/소분류 필터 선적용 ──
    if category_filter:
        base_df = base_df[base_df["equipment_category"].isin(category_filter)]
    if object_type_filter:
        base_df = base_df[base_df["object_type"].isin(object_type_filter)]
    if catalog_filter:
        base_df = base_df[base_df["catalog_profile"].isin(catalog_filter)]
    # ── 2. 퍼지 매칭으로 설비 대/중분류 + 설비명 감지 및 필터 적용 ──
    eq_filter  = extract_equipment_filter(query, base_df, category_filter)
    # 사이드바 대/중/소분류가 이미 적용된 경우 퍼지매칭은 추가 좁히기만
    if not (category_filter or object_type_filter or catalog_filter):
        base_df = apply_equipment_filter(base_df, eq_filter, None)
    else:
        # 사이드바 필터 적용 상태에서 퍼지매칭 소분류/설비명만 추가 적용
        if eq_filter.get("catalog_profiles") and not catalog_filter:
            base_df = base_df[base_df["catalog_profile"].isin(eq_filter["catalog_profiles"])]
        elif eq_filter.get("object_types") and not object_type_filter and not catalog_filter:
            base_df = base_df[base_df["object_type"].isin(eq_filter["object_types"])]
    fuzzy_note = eq_filter.get("applied", "")  # 퍼지매칭 적용 내역

    # 적용된 필터 내역 정리 (프롬프트에 전달)
    applied_notes = []
    if object_type_filter:
        applied_notes.append(f"사이드바 중분류={object_type_filter}")
    if catalog_filter:
        applied_notes.append(f"사이드바 소분류={catalog_filter}")
    if category_filter:
        applied_notes.append(f"사이드바 대분류={category_filter}")
    elif eq_filter.get("category"):
        applied_notes.append(f"대분류={eq_filter['category']}")
    if eq_filter.get("object_types"):
        applied_notes.append(f"중분류≈{eq_filter['object_types']}")
    if eq_filter.get("equipment_descs"):
        applied_notes.append(f"설비명≈{eq_filter['equipment_descs'][:2]}")
    if date_from or date_to:
        df_str = str(date_from) if date_from else "전체시작"
        dt_str = str(date_to)   if date_to   else "전체끝"
        applied_notes.append(f"기간={df_str}~{dt_str}")

    applied_range = ""
    if applied_notes:
        applied_range = (f"\n⚠️ 서버에서 이미 적용된 필터: {', '.join(applied_notes)}"
                         f"\n   → filter_expr에 equipment_category/기간 조건 중복 추가 금지"
                         f"\n   → 추가 텍스트 조건이 없으면 filter_expr = 'df'")

    # summary에 포함할 설비 필터 요약 (기간 제외, 설비 분류만)
    filter_note_parts = []
    if category_filter:
        filter_note_parts.append(", ".join(category_filter))
    elif eq_filter.get("category"):
        filter_note_parts.append(eq_filter["category"])
    if eq_filter.get("object_types"):
        filter_note_parts.append(" · ".join(eq_filter["object_types"][:2]))
    if eq_filter.get("catalog_profiles"):
        filter_note_parts.append(" · ".join(eq_filter["catalog_profiles"][:2]))
    _filter_note = " > ".join(filter_note_parts)  # 예: "크레인 > LIFTING MAGNET 크레인"

    system_prompt = build_system_prompt(base_df) + applied_range

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f'질의: "{query}"'},
            ],
            temperature=0.0, max_tokens=900,
        )
    except Exception as e:
        return {"error": f"OpenAI API 오류: {str(e)}"}

    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"AI 응답 파싱 실패:\n{raw[:300]}"}

    result_type = parsed.get("result_type", "list")
    filter_expr = parsed.get("filter_expr", "df").strip()
    if filter_expr and not filter_expr.startswith("df"):
        filter_expr = f"df[{filter_expr}]"

    safe_bi    = {"len": len, "int": int, "float": float, "round": round, "abs": abs}
    eval_g     = {"df": base_df, "pd": pd, "__builtins__": safe_bi}

    try:
        filtered = eval(filter_expr, eval_g)
        if not isinstance(filtered, pd.DataFrame): raise ValueError("DataFrame 아님")
    except Exception as e:
        return {"error": f"필터 실행 오류: {e}\n코드: {filter_expr}"}

    # ── COUNT ───────────────────────────────────────────
    if result_type == "count":
        agg_col  = parsed.get("agg_col", "")
        agg_func = parsed.get("agg_func", "count")
        try:
            if agg_func == "sum" and agg_col and agg_col in base_df.columns:
                # sum: 텍스트 조건이 있을 수 있으므로 filtered 사용, 비어있으면 base_df
                val = filtered[agg_col].sum() if not filtered.empty else base_df[agg_col].sum()
            elif agg_func == "mean" and agg_col and agg_col in base_df.columns:
                val = filtered[agg_col].mean() if not filtered.empty else base_df[agg_col].mean()
            else:
                # 단순 건수: 퍼지매칭으로 이미 좁혀진 base_df를 항상 기준으로 사용
                # AI의 filter_expr이 잘못된 조건을 추가해도 영향받지 않음
                val = len(base_df)
        except Exception as e:
            return {"error": f"집계 오류: {e}"}
        val_str = f"{val:,.1f}" if isinstance(val, float) else f"{int(val):,}"

        # AI summary는 PLACEHOLDER 미사용 시 잘못된 숫자를 포함할 수 있으므로
        # query_intent만 참조하고 summary는 서버에서 직접 생성
        intent  = parsed.get("query_intent", "")
        ai_tmpl = parsed.get("summary", "")
        scope = f"[{_filter_note}] " if _filter_note else ""
        if "PLACEHOLDER_COUNT" in ai_tmpl or "PLACEHOLDER_VAL" in ai_tmpl:
            summary = (ai_tmpl
                       .replace("PLACEHOLDER_COUNT", val_str)
                       .replace("PLACEHOLDER_VAL",   val_str)
                       .replace("PLACEHOLDER_TOP_NAME", "").replace("PLACEHOLDER_TOP_VAL", ""))
            if scope and not summary.startswith(scope):
                summary = scope + summary
        else:
            unit    = "시간" if agg_func in ("sum","mean") and "duration" in agg_col else "건"
            summary = f"{scope}{val_str}{unit}입니다."

        # 목록: base_df 전체 (퍼지매칭 결과 그대로)
        return {"result_type":"count","count_val":val,"count_str":val_str,
                "records":base_df.head(limit),"agg_result":pd.DataFrame(),
                "total":len(base_df),"summary":summary,
                "query_intent":intent,
                "filter_code":f"[퍼지매칭] {fuzzy_note}\n[AI필터] {filter_expr}","groupby_code":f"{agg_func}→{val_str}","error":None}

    # ── AGGREGATE ───────────────────────────────────────
    # 컬럼 계층: 대분류 → 중분류 → 소분류 → 설비명
    COL_HIERARCHY = [
        ("equipment_category", "대분류"),
        ("object_type",        "중분류"),
        ("catalog_profile",    "소분류"),
        ("equipment_desc",     "설비명"),
    ]
    COL_PARENTS = {
        "object_type":     ["equipment_category"],
        "catalog_profile": ["equipment_category", "object_type"],
        "equipment_desc":  ["equipment_category", "object_type", "catalog_profile"],
    }

    if result_type == "aggregate":
        gc   = parsed.get("groupby_col","equipment_category")
        ac   = parsed.get("agg_col","description")
        af   = parsed.get("agg_func","count")
        top_n= int(parsed.get("top_n",10))
        if gc not in filtered.columns: gc = "equipment_category"
        if ac not in filtered.columns: ac = "description"

        # groupby 시 상위 분류 컬럼도 함께 포함
        parent_cols = [c for c in COL_PARENTS.get(gc, []) if c in filtered.columns]
        group_cols  = parent_cols + [gc]  # 예: ["equipment_category", "object_type"]

        try:
            if af == "count":
                agg = (filtered.groupby(group_cols).size()
                       .reset_index(name="고장건수")
                       .sort_values("고장건수", ascending=False).head(top_n))
                vc  = "고장건수"
            elif af == "sum":
                lbl = "총수리시간(h)" if "duration" in ac else "합계"
                agg = (filtered.groupby(group_cols)[ac].sum()
                       .reset_index(name=lbl)
                       .sort_values(lbl, ascending=False).head(top_n))
                agg[lbl] = agg[lbl].round(1); vc = lbl
            elif af == "mean":
                lbl = "평균수리시간(h)" if "duration" in ac else "평균"
                agg = (filtered.groupby(group_cols)[ac].mean()
                       .reset_index(name=lbl)
                       .sort_values(lbl, ascending=False).head(top_n))
                agg[lbl] = agg[lbl].round(1); vc = lbl
            else: return {"error": f"지원하지 않는 agg_func: {af}"}

            # y축 레이블: 상위분류 > 현재분류 형태로 합성
            if parent_cols:
                sep = " > "
                label_parts = [agg[c].astype(str) for c in group_cols]
                agg["_label"] = label_parts[0]
                for part in label_parts[1:]:
                    agg["_label"] = agg["_label"] + sep + part
                # _label을 첫 컬럼으로 이동, 원래 분류 컬럼들은 제거
                val_cols = [c for c in agg.columns if c not in group_cols and c != "_label"]
                agg = agg[["_label"] + val_cols].rename(columns={"_label": gc})

        except Exception as e: return {"error": f"집계 오류: {e}"}
        total = len(filtered); tmpl = parsed.get("summary","")
        scope = f"[{_filter_note}] " if _filter_note else ""
        if not agg.empty:
            tn = str(agg.iloc[0,0]); tv = agg.iloc[0][vc]
            tvs = f"{tv:,.1f}" if isinstance(tv,float) else f"{int(tv):,}"
            summary = tmpl.replace("PLACEHOLDER_TOP_NAME",tn).replace("PLACEHOLDER_TOP_VAL",tvs).replace("PLACEHOLDER_COUNT",f"{total:,}").replace("PLACEHOLDER_VAL",tvs)
        else:
            summary = tmpl.replace("PLACEHOLDER_COUNT",f"{total:,}").replace("PLACEHOLDER_VAL","")
        if scope and not summary.startswith(scope):
            summary = scope + summary
        return {"result_type":"aggregate","records":pd.DataFrame(),"agg_result":agg,"val_col":vc,
                "total":total,"summary":summary,"query_intent":parsed.get("query_intent",""),
                "filter_code":filter_expr,"groupby_code":f'groupby("{gc}")["{ac}"].{af}() top {top_n}',"error":None}

    # ── FREQ (단일) ──────────────────────────────────────
    # 질의에 "상세내역/상세내용/내용 상/기록 상" 등이 있으면 description 컬럼 강제 사용
    DESC_TRIGGERS = ["상세내역", "상세내용", "내용 상", "기록 상", "설명 상",
                     "description", "텍스트", "자유기재", "기재 내용"]
    _use_desc = any(t in query for t in DESC_TRIGGERS)

    if result_type == "freq":
        hint       = "상세내용" if _use_desc else parsed.get("freq_col","고장원인")
        top_n      = int(parsed.get("top_n", 20))
        total      = len(filtered)
        actual     = _resolve_col(hint, filtered)
        label      = hint if hint in FREQ_COL_MAP else actual
        null_count = int(filtered[actual].isna().sum()) if actual in filtered.columns else 0

        unclassified_info = None  # 미분류 정보 (description 분석 시만)
        if _use_desc and actual == "description":
            if any(w in query for w in ["원인","이유","왜"]):
                aspect = "고장원인"
            elif any(w in query for w in ["현상","증상","나타"]):
                aspect = "고장현상"
            elif any(w in query for w in ["부위","부품","어디"]):
                aspect = "고장부위"
            elif any(w in query for w in ["조치","수리","처리"]):
                aspect = "고장조치"
            else:
                aspect = "종합"
            label    = f"상세내용({aspect})"
            ana_result = _analyze_description_with_ai(
                filtered["description"], client, model, aspect=aspect, top_n=top_n
            )
            kw_df            = ana_result["df"]
            null_count       = 0
            unclassified_info = {
                "classified":   ana_result["classified"],
                "unclassified": ana_result["unclassified"],
                "samples":      ana_result["unclassified_samples"],
            }
        else:
            kw_df = (_value_counts_df(filtered[actual], top_n, label)
                     if actual != "description"
                     else _keyword_df(filtered[actual], top_n))

        # summary: description 분석 시 분류/미분류 건수 포함
        if unclassified_info:
            cl  = unclassified_info["classified"]
            ucl = unclassified_info["unclassified"]
            scope = f"[{_filter_note}] " if _filter_note else ""
            top5  = " · ".join(kw_df["항목"].head(5).tolist()) if not kw_df.empty else "-"
            top1_nm = kw_df.iloc[0]["항목"] if not kw_df.empty else "-"
            top1_v  = int(kw_df.iloc[0]["빈도"]) if not kw_df.empty else 0
            pct     = round(top1_v / cl * 100, 1) if cl > 0 else 0
            summary = (
                f"{scope}총 {total:,}건 중 {label} 분류 {cl:,}건 / 미분류 {ucl:,}건\n\n"
                f"주요 {label}: [{top5}] 순. 1위 '{top1_nm}' {top1_v:,}건({pct}%)"
            )
        else:
            summary = _freq_summary(kw_df, total, label, show_total=True,
                                    null_count=null_count, filter_note=_filter_note)

        return {"result_type":"freq","records":pd.DataFrame(),"agg_result":kw_df,
                "val_col":"빈도","freq_label":label,"freq_results":None,
                "unclassified_info": unclassified_info,
                "total":total,"summary":summary,"query_intent":parsed.get("query_intent",""),
                "filter_code":f"[퍼지매칭] {fuzzy_note}\n[AI필터] {filter_expr}",
                "groupby_code":f"{actual}.exclusive_contains() top {top_n}","error":None}

    # ── MULTI_FREQ (복수 동시 분석) ──────────────────────
    if result_type == "multi_freq":
        hints   = (["상세내용"] if _use_desc
                   else parsed.get("freq_cols", ["고장원인","고장조치"]))
        top_n   = int(parsed.get("top_n", 15))
        total   = len(filtered)

        freq_results = {}
        summaries    = []
        for hint in hints:
            actual = _resolve_col(hint, filtered)
            label  = hint if hint in FREQ_COL_MAP else actual
            kw_df  = (_value_counts_df(filtered[actual], top_n, label)
                      if actual != "description"
                      else _keyword_df(filtered[actual], top_n))
            freq_results[label] = kw_df
            nc = int(filtered[actual].isna().sum()) if actual in filtered.columns else 0
            summaries.append(_freq_summary(kw_df, total, label, show_total=False, null_count=nc))

        scope = f"[{_filter_note}] " if _filter_note else ""
        lines = [f"{scope}총 {total:,}건 분석"] + [f"• {s}" for s in summaries]
        summary = "\n\n".join(lines)
        return {"result_type":"multi_freq","records":pd.DataFrame(),"agg_result":pd.DataFrame(),
                "freq_results":freq_results,"val_col":"빈도","freq_label":"",
                "total":total,"summary":summary,"query_intent":parsed.get("query_intent",""),
                "filter_code":f"[퍼지매칭] {fuzzy_note}\n[AI필터] {filter_expr}","groupby_code":str(hints),"error":None}

    # ── CROSS_FREQ: A별 B 교차 분석 ────────────────────
    if result_type == "cross_freq":
        hints   = parsed.get("freq_cols", ["고장현상", "고장조치"])
        top_n   = int(parsed.get("top_n", 5))
        total   = len(filtered)

        if len(hints) < 2:
            return {"error": "cross_freq는 freq_cols에 2개 항목이 필요합니다."}

        group_hint  = hints[0]   # 그룹 기준 (예: 고장현상)
        target_hint = hints[1]   # 분석 대상 (예: 고장조치)
        group_col   = _resolve_col(group_hint,  filtered)
        target_col  = _resolve_col(target_hint, filtered)
        group_label  = group_hint  if group_hint  in FREQ_COL_MAP else group_col
        target_label = target_hint if target_hint in FREQ_COL_MAP else target_col

        # 그룹별 TOP target 집계
        cross_results = {}
        group_counts  = (filtered[group_col].dropna().value_counts().head(top_n))

        for grp_val in group_counts.index:
            sub = filtered[filtered[group_col] == grp_val]
            kw  = _value_counts_df(sub[target_col], top_n=5, label=target_label)
            cross_results[grp_val] = {"count": len(sub), "top": kw}

        # 요약 문장
        lines = [f"총 {total:,}건 | {group_label} TOP {top_n}별 주요 {target_label} 분석"]
        for grp_val, info in cross_results.items():
            if not info["top"].empty:
                top1 = info["top"].iloc[0]
                lines.append(f"• {grp_val}({info['count']:,}건): 주요 {target_label} → {top1[target_label]} {int(top1['빈도']):,}건")
        summary = "\n\n".join(lines)

        return {
            "result_type":   "cross_freq",
            "records":       pd.DataFrame(),
            "agg_result":    pd.DataFrame(),
            "cross_results": cross_results,
            "group_label":   group_label,
            "target_label":  target_label,
            "val_col":       "빈도",
            "freq_results":  None,
            "total":         total,
            "summary":       summary,
            "query_intent":  parsed.get("query_intent", ""),
            "filter_code":   f"[퍼지매칭] {fuzzy_note}\n[AI필터] {filter_expr}",
            "groupby_code":  f"cross: {group_col} × {target_col} top {top_n}",
            "error":         None,
        }

    # ── GROUPED_MULTI_FREQ: 설비분류별 복수 freq 분석 ───
    if result_type == "grouped_multi_freq":
        gc        = parsed.get("groupby_col", "catalog_profile")
        hints     = parsed.get("freq_cols", ["고장현상", "고장원인"])
        top_n_grp = int(parsed.get("top_n", 10))   # 그룹 TOP N
        top_n_frq = 5                               # 각 그룹 내 freq TOP N
        total     = len(filtered)

        if gc not in filtered.columns:
            gc = "catalog_profile"

        # 상위 분류 레이블 합성 (aggregate와 동일 방식)
        parent_cols = [c for c in COL_PARENTS.get(gc, []) if c in filtered.columns]
        group_cols  = parent_cols + [gc]

        # 그룹별 건수 기준 TOP N 그룹 선택
        grp_counts = (filtered.groupby(group_cols).size()
                      .reset_index(name="_cnt")
                      .sort_values("_cnt", ascending=False)
                      .head(top_n_grp))

        # 그룹 레이블 합성
        if parent_cols:
            label_parts = [grp_counts[c].astype(str) for c in group_cols]
            grp_counts["_label"] = label_parts[0]
            for part in label_parts[1:]:
                grp_counts["_label"] = grp_counts["_label"] + " > " + part
        else:
            grp_counts["_label"] = grp_counts[gc].astype(str)

        # 각 그룹별 freq 분석
        grouped_results = {}  # {label: {hint: DataFrame}}
        for _, row in grp_counts.iterrows():
            grp_label = row["_label"]
            # 해당 그룹의 데이터 추출
            mask = pd.Series([True] * len(filtered), index=filtered.index)
            for col in group_cols:
                mask = mask & (filtered[col] == row[col])
            sub = filtered[mask]
            grp_cnt = int(row["_cnt"])

            freq_dict = {}
            for hint in hints:
                actual = _resolve_col(hint, sub)
                label  = hint if hint in FREQ_COL_MAP else actual
                nc     = int(sub[actual].isna().sum()) if actual in sub.columns else 0
                kw_df  = (_value_counts_df(sub[actual], top_n_frq, label)
                          if actual != "description"
                          else _keyword_df(sub[actual], top_n_frq))
                freq_dict[label] = {"df": kw_df, "count": grp_cnt, "null": nc}
            grouped_results[grp_label] = freq_dict

        # 요약 문장 생성 — 전체 건수 + 컬럼별 미입력 건수 포함
        freq_labels = [hint if hint in FREQ_COL_MAP else _resolve_col(hint, filtered) for hint in hints]

        # 전체 데이터 기준 각 분석 컬럼의 미입력 건수 계산
        null_notes = []
        for hint in hints:
            actual = _resolve_col(hint, filtered)
            label  = hint if hint in FREQ_COL_MAP else actual
            if actual in filtered.columns:
                nc    = int(filtered[actual].isna().sum())
                valid = total - nc
                if nc > 0:
                    null_notes.append(f"{label} 입력 {valid:,}건(미입력 {nc:,}건 제외)")
                else:
                    null_notes.append(f"{label} {valid:,}건")

        gc_label_map = {"equipment_category":"대분류","object_type":"중분류","catalog_profile":"소분류"}
        gc_kr  = gc_label_map.get(gc, gc)
        null_str = " / ".join(null_notes)
        scope  = f"[{_filter_note}] " if _filter_note else ""
        summary = (f"{scope}총 {total:,}건 | {gc_kr} TOP {top_n_grp}\n\n"
                   f"• {null_str}")

        return {
            "result_type":     "grouped_multi_freq",
            "records":         pd.DataFrame(),
            "agg_result":      pd.DataFrame(),
            "grouped_results": grouped_results,   # {grp_label: {freq_label: {df, count, null}}}
            "freq_labels":     freq_labels,
            "hints":           hints,
            "groupby_col":     gc,
            "total":           total,
            "summary":         summary,
            "query_intent":    parsed.get("query_intent", ""),
            "filter_code":     f"[퍼지매칭] {fuzzy_note}\n[AI필터] {filter_expr}",
            "groupby_code":    f"groupby({gc}) × {hints}",
            "error":           None,
        }

    # ── LIST ────────────────────────────────────────────
    sort_by  = parsed.get("sort_by",[])
    sort_asc = parsed.get("sort_asc",[True]*len(sort_by))
    vs = [c for c in sort_by if c in filtered.columns]
    if vs: filtered = filtered.sort_values(vs, ascending=sort_asc[:len(vs)])
    total   = len(filtered); records = filtered.head(limit)
    summary = (parsed.get("summary","")
               .replace("PLACEHOLDER_COUNT",f"{total:,}")
               .replace("PLACEHOLDER_VAL","").replace("PLACEHOLDER_TOP_NAME","").replace("PLACEHOLDER_TOP_VAL",""))
    return {"result_type":"list","records":records,"agg_result":pd.DataFrame(),
            "freq_results":None,"total":total,"summary":summary,
            "query_intent":parsed.get("query_intent",""),
            "filter_code":filter_expr,"groupby_code":"","error":None}
