"""
상세내역(description) 자유텍스트 → 고장 부위/현상/원인/조치 AI 분석 모듈

현재: OpenAI API 사용
추후: 사내 AI로 전환 시 _call_llm() 함수만 교체하면 됩니다.

사용법:
    from utils.ai_description_analyzer import analyze_descriptions
    result = analyze_descriptions(series, api_key, aspect="고장현상")

사내 AI 전환 시:
    1. LLM_PROVIDER = "internal" 로 변경
    2. INTERNAL_AI_URL, INTERNAL_AI_KEY 설정
    3. _call_llm() 내부 로직만 교체 (나머지 코드 동일)
"""

import os
import re
import json
import pandas as pd
from collections import Counter
from typing import Literal

# ── LLM 제공자 설정 ─────────────────────────────────────
# "openai" | "internal" 중 하나로 설정
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "openai")
INTERNAL_AI_URL  = os.getenv("INTERNAL_AI_URL", "http://내부AI서버주소/v1/chat/completions")
INTERNAL_AI_KEY  = os.getenv("INTERNAL_AI_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ASPECT_TYPE = Literal["고장현상", "고장원인", "고장부위", "고장조치", "종합"]

# ── Aspect별 프롬프트 설명 ───────────────────────────────
ASPECT_DESC = {
    "고장현상": "고장 증상/현상 (작동불량·정지·파손·이상·소음·진동·단선 등)",
    "고장원인": "고장 발생 원인 (노후·마모·과부하·접촉불량·사용자부주의·단선 등)",
    "고장부위": "고장이 발생한 부위/부품명 (인버터·베어링·호이스트·유압호스·모터 등)",
    "고장조치": "수행한 조치 내용 (교체·보수·점검·조정·청소·복귀·결선 등)",
    "종합":     "고장과 관련된 핵심 키워드 (현상·원인·부품 포함)",
}


# ── LLM 호출 함수 (전환 대상) ────────────────────────────
def _call_llm(prompt: str, api_key: str, model: str) -> str:
    """
    LLM API를 호출하고 응답 텍스트를 반환합니다.

    ★ 사내 AI 전환 시 이 함수만 수정하세요 ★

    OpenAI 방식:
        POST https://api.openai.com/v1/chat/completions
        Authorization: Bearer {api_key}

    사내 AI 방식 (예시):
        POST http://내부서버/v1/chat/completions
        Authorization: Bearer {INTERNAL_AI_KEY}
        → 대부분의 사내 LLM 서버가 OpenAI 호환 API를 제공합니다.
        → URL과 키만 바꾸면 됩니다.
    """
    import requests

    if LLM_PROVIDER == "internal":
        url     = INTERNAL_AI_URL
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {INTERNAL_AI_KEY or api_key}",
        }
    else:
        url     = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 600,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── 단일 배치 분석 ───────────────────────────────────────
def _analyze_batch(texts: list[str], aspect: str, api_key: str, model: str) -> list[str]:
    """
    텍스트 배치를 LLM에 전달해 각 텍스트의 핵심 키워드를 추출합니다.
    반환: 키워드 리스트 (각 텍스트당 1~2개)
    """
    desc    = ASPECT_DESC.get(aspect, ASPECT_DESC["종합"])
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    prompt = (
        f"설비 고장 기록 텍스트에서 [{desc}]에 해당하는 핵심 키워드를 추출하세요.\n\n"
        "규칙:\n"
        "- 설비명, 태그번호, 위치정보, 호기번호, 톤수는 제외\n"
        "- 텍스트당 1~2개 핵심 키워드만 추출 (해당 없으면 생략)\n"
        "- 유사 표현은 대표 키워드로 통일 (예: 작동불가→작동불능, 동작불량→작동불량)\n"
        "- 반드시 JSON 배열만 반환. 설명 없이.\n"
        '  예: ["작동불량", "FAULT", "마모", "교체"]\n\n'
        f"텍스트:\n{numbered}"
    )

    try:
        raw = _call_llm(prompt, api_key, model)
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if k and 2 <= len(str(k).strip()) <= 20]
    except Exception as e:
        pass
    return []


# ── 메인 분석 함수 ───────────────────────────────────────
def analyze_descriptions(
    series: pd.Series,
    api_key: str,
    aspect: ASPECT_TYPE = "고장현상",
    top_n: int = 20,
    batch_size: int = 50,
    max_records: int = 500,
    model: str = OPENAI_MODEL,
) -> dict:
    """
    description 컬럼 전체를 AI로 분석해 aspect별 키워드 빈도를 반환합니다.

    Args:
        series:      description 컬럼 (pd.Series)
        api_key:     LLM API 키 (사내 AI 전환 시 무시될 수 있음)
        aspect:      분석 관점 ("고장현상"|"고장원인"|"고장부위"|"고장조치"|"종합")
        top_n:       상위 N개 반환
        batch_size:  배치당 텍스트 수 (기본 50)
        max_records: 최대 분석 건수 (비용/속도 제한, 기본 500)
        model:       사용할 모델명

    Returns:
        {
            "df":                 pd.DataFrame(항목, 빈도),
            "total":              int,   # 전체 건수
            "analyzed":           int,   # 실제 분석한 건수 (max_records 제한)
            "classified":         int,   # 키워드 추출된 건수
            "top_keywords":       [str], # TOP 5 키워드
            "error":              str | None,
        }
    """
    texts = series.dropna().reset_index(drop=True)
    total = len(texts)

    if total == 0:
        return _empty_result(total, "분석할 데이터가 없습니다.")

    if not api_key and LLM_PROVIDER == "openai":
        return _empty_result(total, "API Key가 없습니다.")

    # max_records 제한
    sample = texts.head(max_records)
    analyzed = len(sample)

    all_keywords: Counter = Counter()
    classified_count = 0

    # 배치 처리
    for i in range(0, analyzed, batch_size):
        batch   = sample.iloc[i:i+batch_size].tolist()
        kws     = _analyze_batch(batch, aspect, api_key, model)
        if kws:
            all_keywords.update(kws)
            classified_count += len(batch)  # 배치 단위로 집계

    if not all_keywords:
        return _empty_result(total, f"키워드 추출 결과가 없습니다. (분석 {analyzed}건)")

    df_result = pd.DataFrame(
        all_keywords.most_common(top_n),
        columns=["항목", "빈도"]
    )

    # 빈도를 전체 건수 대비 비율로 보정 (샘플링 시)
    if analyzed < total:
        scale = total / analyzed
        df_result["빈도"] = (df_result["빈도"] * scale).round(0).astype(int)

    top5 = df_result["항목"].head(5).tolist()

    return {
        "df":           df_result,
        "total":        total,
        "analyzed":     analyzed,
        "classified":   classified_count,
        "top_keywords": top5,
        "error":        None,
    }


def _empty_result(total: int, msg: str) -> dict:
    return {
        "df":           pd.DataFrame(columns=["항목", "빈도"]),
        "total":        total,
        "analyzed":     0,
        "classified":   0,
        "top_keywords": [],
        "error":        msg,
    }


# ── 직접 실행 시 테스트 ──────────────────────────────────
if __name__ == "__main__":
    """
    사용 예시:
        python utils/ai_description_analyzer.py

    사내 AI 전환 테스트:
        LLM_PROVIDER=internal INTERNAL_AI_URL=http://... python utils/ai_description_analyzer.py
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 테스트 데이터
    test_texts = pd.Series([
        "30TON LMEOC(36-2)주행 동작불량 보수",
        "인버터 폴트 발생",
        "BEARING 마모로 인한 소음 발생",
        "선회작동 불능 점검요망",
        "유압호스 터짐으로 누유 발생",
        "리모콘 전원 안들어옴 보수",
        "Fault 84 발생으로 점검 실시",
        "Motor Trip 발생 보수작업",
        "케이블 소손으로 인한 작동불량",
        "브레이크 마모 교체 작업",
    ])

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️  OPENAI_API_KEY 환경변수가 없습니다.")
        print("   테스트: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"분석 대상: {len(test_texts)}건\n")

    for aspect in ["고장현상", "고장원인", "고장부위", "고장조치"]:
        print(f"=== {aspect} ===")
        result = analyze_descriptions(test_texts, api_key, aspect=aspect, top_n=10, batch_size=5)
        if result["error"]:
            print(f"  오류: {result['error']}")
        else:
            print(f"  분석: {result['analyzed']}건")
            print(f"  TOP5: {result['top_keywords']}")
            print(result["df"].to_string(index=False))
        print()
