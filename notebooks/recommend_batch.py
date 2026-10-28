"""7주차 산출물 — 실행 한 번으로 전원의 추천 결과 파일을 만든다.

    python recommend_batch.py

하는 일 (6주차까지 노트북에서 셀을 하나씩 눌러 하던 일을 그대로 묶었다) — 파이프라인(pipeline)
  ① 들어온 데이터에 빈 값 · 중복 · 없는 번호가 없는지 검사한다   — 데이터 점검
  ② 거래 기록 전체로 모델을 학습한다                            — 학습(training)
  ③ 300명 모두에게 위험도 상한을 지킨 추천 10개                 — 일괄 추천(batch recommendation)
  ④ 결과 표가 정해 둔 규격을 지켰는지 검사한다                  — 규격 검사
  ⑤ 통과하면 CSV 파일 하나에 저장한다                           — 결과 파일
  ① 이나 ④ 에서 문제가 나오면 그 자리에서 멈춘다. 잘못된 파일을 남기지 않는다.

다음 주 API(FastAPI)는 이 파일을 읽기만 한다 — 서빙(serving). API 는 모델을 다시 학습하지 않는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import recsys

출력_파일 = Path("output") / "recommendations.csv"       # 결과 파일이 놓일 자리
상한 = {"안정형": 3, "성장형": 4, "공격형": 5}          # 6주차 적합 기준 — 투자성향별 최대 위험도
규격_열 = ["user_id", "rank", "item_id", "item_name", "risk_level", "persona", "score"]


def 데이터_점검(items, users, 기록):
    """들어온 데이터를 검사해 {검사 이름: 문제 개수} 를 돌려준다. 모두 0 이면 통과."""
    return {"빈 값": int(items.isna().sum().sum() + users.isna().sum().sum() + 기록.isna().sum().sum()),
            "똑같은 거래 기록": int(기록.duplicated().sum()),
            "종목 목록에 없는 종목 번호": int((~기록["item_id"].isin(items["item_id"])).sum()),
            "투자자 목록에 없는 투자자 번호": int((~기록["user_id"].isin(users["user_id"])).sum()),
            "세 가지가 아닌 투자성향": int((~users["persona"].isin(상한)).sum()),
            "1 ~ 5 를 벗어난 위험도": int((~items["risk_level"].between(1, 5)).sum())}


def 학습하기(기록, items):
    """거래 기록으로 TruncatedSVD 를 학습하고, 서빙에 필요한 것을 딕셔너리 하나에 담는다."""
    상호작용_행렬 = recsys.interaction_matrix(기록, items)
    모델 = TruncatedSVD(n_components=8, random_state=42)    # random_state 고정 = 다시 돌려도 같은 결과
    return {"사용자_요인": 모델.fit_transform(상호작용_행렬),
            "아이템_요인": 모델.components_,
            "사용자_목록": list(상호작용_행렬.index),
            "아이템_목록": list(상호작용_행렬.columns),
            "인기순위": list(기록["item_id"].value_counts().index),
            "이미_담은것": 기록.groupby("user_id")["item_id"].apply(set).to_dict()}


def 추천하기(묶음, 사람, 성향, 위험도):
    """한 사람에게 위험도 상한을 지킨 추천 10개를 (종목 번호, 점수) 목록으로 준다."""
    점수_딕셔너리 = {}
    if 사람 in 묶음["사용자_목록"]:
        점수 = 묶음["사용자_요인"][묶음["사용자_목록"].index(사람)] @ 묶음["아이템_요인"]
        순서 = []
        for 자리 in np.argsort(-점수):
            순서.append(묶음["아이템_목록"][자리])
            점수_딕셔너리[묶음["아이템_목록"][자리]] = float(점수[자리])
    else:
        순서 = 묶음["인기순위"]                              # 기록이 없는 사람은 인기 순서
    통과 = []
    for 번호 in 순서:
        if 위험도[번호] <= 상한[성향[사람]]:
            통과.append(번호)
    결과 = []
    for 번호 in recsys.take(통과, 묶음["이미_담은것"].get(사람, set())):
        결과.append((번호, round(점수_딕셔너리.get(번호, 0.0), 4)))
    return 결과


def 일괄_추천(묶음, users, items):
    """300명 모두의 추천을 규격대로 표 하나에 담는다."""
    종목 = items.set_index("item_id")
    성향 = users.set_index("user_id")["persona"].to_dict()
    위험도 = 종목["risk_level"].to_dict()
    줄들 = []
    for 사람 in users["user_id"]:
        순위 = 1
        for 번호, 점수 in 추천하기(묶음, 사람, 성향, 위험도):
            줄들.append({"user_id": 사람, "rank": 순위, "item_id": 번호,
                         "item_name": 종목.loc[번호, "name"], "risk_level": int(위험도[번호]),
                         "persona": 성향[사람], "score": 점수})
            순위 = 순위 + 1
    return pd.DataFrame(줄들, columns=규격_열)


def 규격_검사(표, users, 묶음):
    """결과 표가 규격을 지켰는지 검사하고, 어긴 것을 목록으로 돌려준다. 빈 목록이면 통과."""
    문제 = []
    if list(표.columns) != 규격_열:
        문제.append("열 이름이나 순서가 규격과 다릅니다")
    if 표["user_id"].nunique() != len(users):
        문제.append("추천이 없는 사람이 있습니다")
    if (표.groupby("user_id").size() != 10).any():
        문제.append("추천이 10개가 아닌 사람이 있습니다")
    for _, 줄 in 표.iterrows():
        if 줄["risk_level"] > 상한[줄["persona"]]:
            문제.append(f"{줄['user_id']} 에게 상한을 넘는 {줄['item_id']}")
        if 줄["item_id"] in 묶음["이미_담은것"].get(줄["user_id"], set()):
            문제.append(f"{줄['user_id']} 에게 이미 담은 {줄['item_id']}")
    return 문제


def main():
    items, users, interactions = recsys.load()

    점검 = 데이터_점검(items, users, interactions)
    if sum(점검.values()) > 0:
        raise SystemExit(f"① 데이터 점검 실패 — {점검}")
    print(f"① 데이터 점검 — 여섯 가지 모두 0건 · 통과")

    묶음 = 학습하기(interactions, items)                    # 서비스용은 거래 기록 전체로 학습
    print(f"② 학습 — 거래 기록 {len(interactions):,}건 · 투자자 {len(묶음['사용자_목록'])}명")

    표 = 일괄_추천(묶음, users, items)
    print(f"③ 일괄 추천 — {len(표):,}줄")

    문제 = 규격_검사(표, users, 묶음)
    if 문제:
        raise SystemExit("④ 규격 검사 실패 — " + " / ".join(문제[:5]))
    print("④ 규격 검사 — 문제 0개 · 통과")

    출력_파일.parent.mkdir(exist_ok=True)
    표.to_csv(출력_파일, index=False, encoding="utf-8-sig")  # utf-8-sig = 엑셀에서 한글이 안 깨진다
    print(f"⑤ 저장 — {출력_파일} · {표['user_id'].nunique()}명 × 10개")


if __name__ == "__main__":
    main()
