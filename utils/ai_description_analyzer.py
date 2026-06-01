"""
상세내역(description) 자유텍스트 → 고장 부위/현상/원인/조치 AI 분석 모듈

현재: LangChain + OpenAI API 사용
사내 AI 전환 시: .env에 OPENAI_BASE_URL 추가만 하면 됩니다.

사용법:
    from utils.ai_description_analyzer import analyze_descriptions
    result = analyze_descriptions(series, api_key, aspect="고장현상")

사내 AI 전환 시 .env 설정:
    OPENAI_API_KEY=사내AI키
    OPENAI_MODEL=사내모델명
    OPENAI_BASE_URL=http://사내서버/v1   ← 이것만 추가
"""

import os
import json
import pandas as pd
from collections import Counter
from typing import Literal
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ASPECT_TYPE = Literal["고장현상", "고장원인", "고장부위", "고장조치", "종합"]

ASPECT_DESC = {
    "고장현상": "고장 증상/현상 (작동불량·정지·파손·이상·소음·진동·단선 등)",
    "고장원인": "고장 발생 원인 (노후·마모·과부하·접촉불량·사용자부주의·단선 등)",
    "고장부위": "고장이 발생한 부위/부품명 (인버터·베어링·호이스트·유압호스·모터 등)",
    "고장조치": "수행한 조치 내용 (교체·보수·점검·조정·청소·복귀·결선 등)",
    "종합":     "고장과 관련된 핵심 키워드 (현상·원인·부품 포함)",
}


def _call_llm(prompt: str, api_key: str, model: str) -> str:
    """
    LLM API를 호출하고 응답 텍스트를 반환합니다.
    LangChain v1 init_chat_model 사용.

    ★ 사내 AI 전환 시 이 함수만 수정하세요 ★
    .env에 OPENAI_BASE_URL 추가만 하면 사내 서버로 전환됩니다.
    """
    llm = init_chat_model(
        model=f"openai:{model}",
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.0,
        max_tokens=600,
    )
    resp: AIMessage = llm.invoke([("human", prompt)])  # v1 튜플 형식
    return resp.content.strip()


def _analyze_batch(texts: list, aspect: str, api_key: str, model: str) -> list:
    """텍스트 배치를 LLM에 전달해 각 텍스트의 핵심 키워드를 추출합니다."""
    desc     = ASPECT_DESC.get(aspect, ASPECT_DESC["종합"])
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
        raw   = _call_llm(prompt, api_key, model)
        raw   = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if k and 2 <= len(str(k).strip()) <= 20]
    except Exception:
        pass
    return []


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
        api_key:     LLM API 키
        aspect:      분석 관점 ("고장현상"|"고장원인"|"고장부위"|"고장조치"|"종합")
        top_n:       상위 N개 반환
        batch_size:  배치당 텍스트 수 (기본 50)
        max_records: 최대 분석 건수 (기본 500)
        model:       사용할 모델명

    Returns:
        {
            "df":           pd.DataFrame(항목, 빈도),
            "total":        int,
            "analyzed":     int,
            "classified":   int,
            "top_keywords": [str],
            "error":        str | None,
        }
    """
    texts = series.dropna().reset_index(drop=True)
    total = len(texts)

    if total == 0:
        return _empty_result(total, "분석할 데이터가 없습니다.")
    if not api_key:
        return _empty_result(total, "API Key가 없습니다. .env 파일의 OPENAI_API_KEY를 확인하세요.")

    sample   = texts.head(max_records)
    analyzed = len(sample)

    all_keywords: Counter = Counter()
    classified_count = 0

    for i in range(0, analyzed, batch_size):
        batch = sample.iloc[i:i+batch_size].tolist()
        kws   = _analyze_batch(batch, aspect, api_key, model)
        if kws:
            all_keywords.update(kws)
            classified_count += len(batch)

    if not all_keywords:
        return _empty_result(total, f"키워드 추출 결과가 없습니다. (분석 {analyzed}건)")

    df_result = pd.DataFrame(
        all_keywords.most_common(top_n),
        columns=["항목", "빈도"]
    )

    # 샘플링 시 전체 비율로 보정
    if analyzed < total:
        scale = total / analyzed
        df_result["빈도"] = (df_result["빈도"] * scale).round(0).astype(int)

    return {
        "df":           df_result,
        "total":        total,
        "analyzed":     analyzed,
        "classified":   classified_count,
        "top_keywords": df_result["항목"].head(5).tolist(),
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


if __name__ == "__main__":
    """
    테스트 실행:
        python utils/ai_description_analyzer.py

    사내 AI 전환 테스트:
        OPENAI_BASE_URL=http://사내서버/v1 python utils/ai_description_analyzer.py
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_texts = pd.Series([
        "30TON LMEOC(36-2)주행 동작불량 보수",
        "인버터 폴트 발생",
        "BEARING 마모로 인한 소음 발생",
        "선회작동 불능 점검요망",
        "유압호스 터짐으로 누유 발생",
        "리모콘 전원 안들어옴 보수",
        "Fault 84 발생으로 점검 실시",
        "Motor Trip 발생 보수작업",
    ])

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("⚠️  .env 파일에 OPENAI_API_KEY를 설정하세요.")
        sys.exit(1)

    print(f"모델: {OPENAI_MODEL}")
    print(f"분석 대상: {len(test_texts)}건\n")

    for aspect in ["고장현상", "고장원인"]:
        print(f"=== {aspect} ===")
        result = analyze_descriptions(test_texts, api_key, aspect=aspect, top_n=5, batch_size=4)
        if result["error"]:
            print(f"  오류: {result['error']}")
        else:
            print(f"  TOP5: {result['top_keywords']}")
        print()
