"""8주차 산출물 — 7주차 결과 파일을 읽어, 투자자 번호를 받으면 추천을 돌려주는 API.

    python -m uvicorn recommend_api:app --port 8765

엔드포인트(요청을 받는 주소) 두 개
  GET /health                      API 가 켜져 있는지 확인
  GET /recommend/{user_id}?k=10    그 투자자의 추천 k개 (k 는 1 ~ 10, 안 적으면 10)

API 는 모델을 학습하지 않는다. 켜질 때 결과 파일을 한 번 읽고, 요청이 오면 꺼내 주기만 한다 — 서빙(serving).
결과 파일은 7주차 recommend_batch.py 가 만든다. 파일이 새로 만들어지면 API 를 다시 켜야 새 추천이 반영된다.
"""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

결과_파일 = Path(__file__).resolve().parent / "output" / "recommendations.csv"
추천표 = pd.read_csv(결과_파일, encoding="utf-8-sig")      # 켜질 때 한 번만 읽는다

app = FastAPI(title="추천 API",
              description="7주차 결과 파일(recommendations.csv)을 읽어 투자자별 추천을 돌려줍니다.")


@app.get("/health")
def health():
    """API 가 켜져 있는지, 결과 파일을 몇 줄 읽었는지 알려 준다."""
    return {"status": "ok", "rows": len(추천표)}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=10, description="몇 개를 받을지 (1 ~ 10)")):
    """투자자 번호를 받아, 위험도 상한을 지킨 추천을 순위대로 k개 돌려준다."""
    줄들 = 추천표[추천표["user_id"] == user_id].sort_values("rank").head(k)
    if 줄들.empty:                                              # 결과 파일에 없는 번호
        raise HTTPException(status_code=404, detail=f"{user_id} 는 추천이 없는 투자자입니다")
    추천 = []
    for _, 줄 in 줄들.iterrows():
        추천.append({"rank": int(줄["rank"]), "item_id": 줄["item_id"], "item_name": 줄["item_name"],
                     "risk_level": int(줄["risk_level"]), "score": float(줄["score"])})
    return {"user_id": user_id, "persona": 줄들["persona"].iloc[0], "recommendations": 추천}
