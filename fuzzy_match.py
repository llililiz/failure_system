"""
퍼지 매칭 유틸리티
매칭 우선순위: 소분류(catalog_profile) → 중분류(object_type) → 대분류(equipment_category) → 설비명
더 구체적인 매칭이 있으면 상위 분류보다 우선 적용.
"""

import re
import unicodedata
from typing import Optional


# ── 정규화 ──────────────────────────────────────────────
def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r'[\s/\-\(\)\[\]_·,\.㎥㎡]+', '', text)


# ── 한글 → 영문/표준형 변환 ──────────────────────────────
# 가능한 한 많은 한글 설비 표현을 영문으로 변환
KO_TO_EN = [
    # 크레인 종류
    (r'앵글크레인',     'anglecrane'),
    (r'앵글',          'angle'),
    (r'오버헤드크레인', 'overheadcrane'),
    (r'오버헤드',       'overhead'),
    (r'갠트리크레인',   'gantrycrane'),
    (r'갠트리',         'gantry'),
    (r'골리아스크레인', 'goliathcrane'),
    (r'골리아스',       'goliath'),
    (r'골리앗',         'goliath'),
    (r'집크레인',       'jibcrane'),
    (r'집형크레인',     'jibcrane'),
    (r'타워크레인',     'towercrane'),
    (r'천장크레인',     'eoc'),
    (r'천장형크레인',   'eoc'),
    (r'호이스트크레인', 'hoistcrane'),
    (r'리프팅마그넷',   'liftingmagnet'),
    (r'리프팅매그넷',   'liftingmagnet'),
    (r'마그넷크레인',   'liftingmagnet'),
    (r'호이스트',       'hoist'),
    (r'크레인',         'crane'),
    # 운반/이송
    (r'트랜스포터',     'transporter'),
    (r'컨베이어',       'conveyor'),
    (r'컨베어',         'conveyor'),
    (r'벨트컨베이어',   'beltconveyor'),
    (r'블라스트벨트',   'blastbelt'),
    (r'블라스트',       'blast'),
    # 승강
    (r'리프트',         'lift'),
    (r'승강기',         'lift'),
    (r'엘리베이터',     'elevator'),
    # AC타워
    (r'ac타워',         'actower'),
    (r'에이씨타워',     'actower'),
    (r'에어컨타워',     'actower'),
    # 용접
    (r'용접기',         'welder'),
    (r'용접장비',       'welding'),
    (r'플라즈마',       'plasma'),
    (r'피엘에스',       'pls'),
    # 프레스/가공
    (r'롤벤딩프레스',   'rollbendingpress'),
    (r'롤벤딩',         'rollbending'),
    (r'프레스',         'press'),
    # 기타
    (r'에어컨',         '냉난방기'),
    (r'냉방',           '냉난방기'),
    (r'펌프',           'pump'),
]

TON_PATTERN = re.compile(r'(\d+)\s*톤')


def normalize_query(query: str) -> str:
    """질의어 정규화: 한글 약어→영문, 정규화"""
    q = query.lower()
    q = TON_PATTERN.sub(lambda m: m.group(1) + 'ton', q)
    for ko, en in KO_TO_EN:
        q = re.sub(ko, en, q, flags=re.IGNORECASE)
    return normalize(q)


# ── 유사도 계산 ──────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    if a == b: return 1.0
    if a in b: return 0.85 + 0.15 * len(a) / max(len(b), 1)
    if b in a: return 0.75 + 0.15 * len(b) / max(len(a), 1)

    def ng(s, n=2): return set(s[i:i+n] for i in range(len(s)-n+1))
    ag, bg = ng(a), ng(b)
    overlap = len(ag & bg) / max(len(ag), len(bg), 1) if ag and bg else 0.0

    mn = min(len(a), len(b))
    prefix = 0.0
    if mn >= 2:
        for i in range(mn):
            if a[i] != b[i]: prefix = i / mn; break
        else: prefix = 1.0

    return round(0.6 * overlap + 0.4 * prefix, 3)


def _best_match(q_norm: str, candidates: list, threshold: float, top_k: int = 5):
    """정규화된 질의와 후보 목록을 비교, (원본값, 점수) 리스트 반환"""
    scored = []
    for orig in candidates:
        c_norm = normalize(orig)
        if not c_norm: continue
        score = _sim(q_norm, c_norm)
        if score >= threshold:
            scored.append((orig, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ── 핵심 함수 ─────────────────────────────────────────────
def extract_equipment_filter(query: str, df, category_filter: list = None) -> dict:
    """
    매칭 우선순위: 소분류 → 중분류 → 대분류 → 설비명
    더 구체적인 매칭이 존재하면 상위 분류보다 우선.
    """
    result = {
        "category":        None,    # 대분류
        "object_types":    [],      # 중분류
        "catalog_profiles": [],     # 소분류
        "equipment_descs": [],      # 설비명
        "desc_keywords":   [],
        "applied":         "",
    }
    applied_parts = []
    q_norm = normalize_query(query)

    # 사이드바 대분류 필터가 있으면 그 범위 안에서만 검색
    base = df.copy()
    if category_filter:
        base = base[base["equipment_category"].isin(category_filter)]

    # ── Step 1: 소분류(catalog_profile) 퍼지 매칭 ──
    cp_list = base["catalog_profile"].dropna().unique().tolist()
    cp_matches = _best_match(q_norm, cp_list, threshold=0.45)
    if cp_matches and cp_matches[0][1] >= 0.5:
        top_score = cp_matches[0][1]
        matched = [m[0] for m in cp_matches if m[1] >= top_score * 0.85]
        result["catalog_profiles"] = matched
        applied_parts.append(f"소분류={matched}")

    # ── Step 2: 중분류(object_type) 퍼지 매칭 ──
    # 소분류보다 점수가 높으면 중분류로 교체, 낮으면 소분류 유지
    ot_list = base["object_type"].dropna().unique().tolist()
    ot_matches = _best_match(q_norm, ot_list, threshold=0.45)
    if ot_matches and ot_matches[0][1] >= 0.55:
        top_ot_score = ot_matches[0][1]
        top_cp_score = cp_matches[0][1] if cp_matches else 0.0
        if top_ot_score > top_cp_score:
            # 중분류가 더 잘 맞음 → 소분류 결과 무효화하고 중분류 사용
            result["catalog_profiles"] = []
            matched = [m[0] for m in ot_matches if m[1] >= top_ot_score * 0.85]
            result["object_types"] = matched
            applied_parts = [p for p in applied_parts if not p.startswith("소분류")]
            applied_parts.append(f"중분류={matched}")

    # ── Step 3: 대분류(equipment_category) 감지 ──
    # 소분류/중분류 미매칭 시, 또는 사이드바 미선택 시 대분류 감지
    if not result["catalog_profiles"] and not result["object_types"] and not category_filter:
        all_cats = base["equipment_category"].dropna().unique().tolist()
        for cat in all_cats:
            if normalize(cat) in q_norm or cat in query:
                result["category"] = cat
                applied_parts.append(f"대분류={cat}")
                break

    # ── Step 4: 설비명 직접 패턴 (소분류/중분류/대분류 모두 미매칭 시) ──
    if not result["catalog_profiles"] and not result["object_types"] and not result["category"]:
        eq_descs = base["equipment_desc"].dropna().unique().tolist()
        DIRECT = [
            ("리프트",  lambda d: "리프트" in str(d)),
            ("lift",    lambda d: "lift"   in str(d).lower()),
            ("goliath", lambda d: "GOLIATH" in str(d).upper()),
            ("gantry",  lambda d: "GANTRY"  in str(d).upper()),
        ]
        for pat, fn in DIRECT:
            if pat in q_norm:
                hits = [d for d in eq_descs if fn(d)]
                if hits:
                    result["equipment_descs"] = hits
                    applied_parts.append(f"설비명={pat}포함 {len(hits)}건")
                    break

    # ── Step 5: 모두 실패 시 description 키워드 보존 ──
    if not any([result["catalog_profiles"], result["object_types"],
                result["category"], result["equipment_descs"]]):
        stopwords = {"주요","고장","원인","조치","현황","결과","증상","부위",
                     "목록","내역","건수","분석","보여","평균","시간","현상"}
        result["desc_keywords"] = [
            k for k in re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}', query)
            if k.lower() not in stopwords
        ]

    result["applied"] = ", ".join(applied_parts)
    return result


def apply_equipment_filter(df, eq_filter: dict, category_filter: list = None):
    dff = df.copy()

    # 대분류 (사이드바 우선, 없으면 질의 감지)
    if category_filter:
        dff = dff[dff["equipment_category"].isin(category_filter)]
    elif eq_filter.get("category"):
        dff = dff[dff["equipment_category"] == eq_filter["category"]]

    # 소분류 / 중분류 (더 높은 점수쪽이 result에 남아있음)
    if eq_filter.get("catalog_profiles"):
        dff = dff[dff["catalog_profile"].isin(eq_filter["catalog_profiles"])]
    elif eq_filter.get("object_types"):
        dff = dff[dff["object_type"].isin(eq_filter["object_types"])]
    elif eq_filter.get("equipment_descs"):
        dff = dff[dff["equipment_desc"].isin(eq_filter["equipment_descs"])]

    return dff
