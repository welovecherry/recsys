"""8주차 산출물 — 7주차 결과 파일을 읽어, 투자자 번호를 받으면 추천을 응답하는 API.

    python -m uvicorn recommend_api:app --port 8765

엔드포인트(요청을 받는 주소) 두 개
  GET /health                      API 가 켜져 있는지 확인
  GET /recommend/{user_id}?k=10    그 투자자의 추천 k개 (k 는 1 ~ 10, 안 적으면 10)

API 는 모델을 학습하지 않는다. 켜질 때 결과 파일을 한 번 읽어 메모리에 두고,
요청이 오면 그 투자자의 줄을 골라 응답하기만 한다 — 서빙(serving).
결과 파일은 7주차 recommend_batch.py 가 만든다. 파일이 새로 만들어지면 API 를 다시 켜야 새 추천이 반영된다.
"""
from pathlib import Path          # 파일 위치(경로)를 다루는 도구

import pandas as pd               # 표를 다루는 도구
from fastapi import FastAPI, HTTPException, Query   # API 를 만드는 도구 · 오류 응답 · 쿼리 매개변수 규칙
from pydantic import BaseModel    # 응답의 모양(항목 이름과 형식)을 정하는 도구. FastAPI 와 함께 설치되어 있다

결과_파일 = Path(__file__).resolve().parent / "output" / "recommendations.csv"   # 7주차가 저장한 파일
필요한_열 = ["user_id", "rank", "item_id", "item_name", "risk_level", "persona", "score"]   # 7주차 결과 파일 규격(output schema)


# ── 켜질 때 한 번만 실행된다 ──────────────────────────────────────────────

if not 결과_파일.exists():                                      # 파일이 없으면 API 를 켜지 않고 이유를 알려 준다
    raise FileNotFoundError(f"결과 파일이 없습니다: {결과_파일}\n7주차의 python recommend_batch.py 를 먼저 실행하세요")

추천표 = pd.read_csv(결과_파일, encoding="utf-8-sig")           # 3,000줄을 메모리에 올려 둔다. 요청마다 파일을 다시 읽지 않는다

if 추천표.empty:                                               # 줄이 하나도 없는 파일이면 멈춘다
    raise ValueError("결과 파일이 비어 있습니다")

빠진_열 = [열 for 열 in 필요한_열 if 열 not in 추천표.columns]
if 빠진_열:                                                    # 규격과 다른 파일이면 잘못된 응답이 나가기 전에 멈춘다
    raise ValueError(f"결과 파일에 필요한 열이 없습니다: {빠진_열}")


# ── 응답의 모양 (response schema) — Swagger 화면에도 그대로 나온다 ──────────

class RecommendItem(BaseModel):          # 영어 이름: 한글 이름은 Swagger 화면에서 깨져 보인다
    rank: int          # 추천 순위 (1 = 가장 먼저 추천)
    item_id: str       # 종목 번호
    item_name: str     # 종목 이름
    risk_level: int    # 종목의 위험도 (1 ~ 5)
    score: float       # 추천 점수


class RecommendResponse(BaseModel):
    user_id: str                       # 투자자 번호
    persona: str                       # 투자 성향
    recommendations: list[RecommendItem]   # 추천 목록 (순위 순서)


# ── 엔드포인트 ────────────────────────────────────────────────────────

app = FastAPI(title="추천 API",
              description="7주차 결과 파일(recommendations.csv)을 읽어 투자자별 추천을 응답합니다.")


@app.get("/health")
def health():
    """API 가 켜져 있는지, 결과 파일을 몇 줄 읽었는지 알려 준다."""
    return {"status": "ok", "rows": len(추천표)}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str,
              k: int = Query(10, ge=1, le=10, description="몇 개를 받을지 (1 ~ 10)")):
    """투자자 번호를 받아, 위험도 상한을 지킨 추천을 순위대로 k개 응답한다."""
    # 1) 그 투자자의 줄만 고른다
    그_투자자 = 추천표[추천표["user_id"] == user_id]

    # 2) 결과 파일에 없는 번호면 404 (찾는 것이 없음) 로 응답하고 끝낸다
    if 그_투자자.empty:
        raise HTTPException(status_code=404, detail=f"{user_id} 는 추천이 없는 투자자입니다")

    # 3) 순위 순서로 정렬해서 위에서 k개만 남긴다
    위에서_k개 = 그_투자자.sort_values("rank").head(k)

    # 4) 한 줄씩 응답의 모양(RecommendItem)에 맞게 만든다
    추천_목록 = []
    for 줄 in 위에서_k개.itertuples():
        추천_목록.append(RecommendItem(
            rank=int(줄.rank),
            item_id=줄.item_id,
            item_name=줄.item_name,
            risk_level=int(줄.risk_level),
            score=float(줄.score),
        ))

    # 5) 투자자 번호 · 투자 성향 · 추천 목록을 응답한다 (FastAPI 가 JSON 으로 바꿔서 보낸다)
    return RecommendResponse(user_id=user_id, persona=위에서_k개["persona"].iloc[0], recommendations=추천_목록)
