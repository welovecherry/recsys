"""가상 투자자 300명과 그들의 거래 기록을 만든다.

왜 가상 데이터인가
------------------
추천시스템은 "누가 무엇을 골랐나" 기록을 먹고 산다. 그런데 주식·ETF 시세 데이터에는
사용자가 없다. 그래서 가상 투자자와 그들의 행동을 직접 만들어 넣어야 한다.

무엇을 심어 두는가 (이게 이 파일의 핵심)
----------------------------------------
강의는 주차마다 모델을 바꿔가며 같은 지표(Recall@10)로 점수를 잰다. 점수가 올라가려면
데이터 안에 난이도가 다른 신호가 층층이 들어 있어야 한다.

    신호 층                          이걸 잡아내는 주차
    -------------------------------  ------------------------------
    전체 인기 종목 (누구나 담는 것)   1주차 · 인기순 추천
    투자 성향별 상품 선호             2주차 · 성향별 인기 추천
    성향 안의 더 잘게 나뉜 취향 그룹  4주차 · 행렬분해(ALS)
    개인 고유의 무작위 선택           아무도 못 잡음 (점수 상한 역할)

성향은 3개(안정형·성장형·공격형)뿐이라 2주차 모델도 금방 잡아낸다. 그래서 성향 안에
하위 취향 그룹을 4개씩, 모두 12개를 더 숨겨 둔다. 성향별 인기 추천으로는 12개를
구분할 수 없고 행렬분해는 구분할 수 있다. 이 격차가 4주차에 점수가 오르는 이유다.

하위 그룹의 취향은 여러 사람이 함께 나눠 가져야 한다. 완전히 개인적인 취향은 어떤
모델도 배울 수 없기 때문이다.

실행
----
    uv run python data/make_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent

# ---------------------------------------------------------------- 튜닝 손잡이
# 점수 사다리(1주차 < 2주차 < 4주차)가 안 만들어지면 여기를 조정한다.
# data/check_data.py 가 사다리를 실제로 재서 보여준다.

N_USERS = 300
N_SUBCLUSTERS_PER_PERSONA = 4          # 성향 3개 × 4 = 잠재 그룹 12개
INTERACTIONS_PER_USER = (25, 75)       # 사용자별 거래 횟수 범위 → 총 7,000건 안팎
SUBCLUSTER_BASKET_SIZE = 12            # 하위 그룹 하나가 즐겨 담는 종목 수
SUBCLUSTER_QUIRKS = 3                  # 그중 성향과 무관한 종목 수 (행렬분해만 잡는 신호)

# 거래 한 건이 어느 신호에서 나오는지의 비율. 합이 1.0이어야 한다.
P_GLOBAL_POPULAR = 0.20                # 1주차가 잡는 신호
P_PERSONA        = 0.22                # 2주차가 잡는 신호
P_SUBCLUSTER     = 0.48                # 4주차 행렬분해가 잡는 신호
P_NOISE          = 0.10                # 아무도 못 잡는 잡음

# 성향에 맞는 상품을 고를 확률이 100%가 되면 추천 문제가 너무 쉬워져서 주차별 차이가
# 안 보인다. 현실처럼 성향 밖 상품도 섞이도록 이 값으로 뾰족한 정도를 조절한다.
PERSONA_SHARPNESS = 2.2

MONTHS = 12                            # 전체 기간
TEST_MONTHS = 1                        # 마지막 1개월이 3주차의 테스트 구간이 된다
NEWCOMER_RATE = 0.05                   # 마지막 달에만 나타나는 신규 투자자 비율
                                       # → 3주차 콜드스타트 시연용

SEED = 42

# ------------------------------------------------------- 성향별로 좋아하는 것
# 은행이 실제로 가지고 있을 법한 상품 정보(섹터·테마·위험도)만 가지고 선호를 만든다.
# items.csv 에 "이 종목은 안정형용" 같은 정답을 적어두지 않는다. 그건 답을 흘리는 것이다.

PERSONA_TASTE = {
    "안정형": {
        "sector": {"배당": 3.0, "채권": 3.0, "리츠": 2.5, "저변동": 2.5, "통신": 2.2,
                   "소비재": 2.0, "유틸리티": 2.0, "금융": 1.8, "헬스케어": 1.5, "지수": 1.0},
        "theme": {"배당": 3.0, "채권": 3.0, "가치": 2.5, "인덱스": 1.2},
        "risk_center": 1.8,            # 이 위험도 근처를 좋아한다
    },
    "성장형": {
        "sector": {"반도체": 3.0, "기술": 3.0, "인터넷플랫폼": 2.8, "소프트웨어": 2.5,
                   "게임": 2.2, "IT서비스": 2.0, "지수": 1.5, "미디어": 1.5},
        "theme": {"성장": 3.0, "인덱스": 1.5, "AI": 2.0},
        "risk_center": 3.6,
    },
    "공격형": {
        "sector": {"레버리지": 3.0, "인버스": 2.5, "2차전지": 3.0, "테마": 2.8,
                   "바이오": 2.5, "방산": 2.2, "친환경": 2.2, "자동차": 1.8},
        "theme": {"레버리지": 3.0, "친환경": 2.5, "AI": 2.5, "테마": 2.5, "성장": 1.2},
        "risk_center": 4.7,
    },
}


def item_scores_for(persona, items):
    """성향 하나가 각 종목을 얼마나 좋아하는지 점수로 매긴다."""
    taste = PERSONA_TASTE[persona]
    sector = items["sector"].map(taste["sector"]).fillna(0.3)
    theme = items["theme"].map(taste["theme"]).fillna(0.3)
    # 위험도가 취향에서 멀수록 점수가 깎인다
    risk_gap = (items["risk_level"] - taste["risk_center"]).abs()
    risk = np.exp(-0.5 * risk_gap)
    return (sector + theme).to_numpy() * risk.to_numpy()


def to_probs(scores, sharpness):
    """점수를 확률로 바꾼다. sharpness 가 클수록 취향이 뾰족해진다."""
    p = np.power(np.clip(scores, 1e-9, None), sharpness)
    return p / p.sum()


def main():
    rng = np.random.default_rng(SEED)
    items = pd.read_csv(DATA / "items.csv")
    n_items = len(items)
    personas = list(PERSONA_TASTE)

    # --- 전체 인기 분포 -----------------------------------------------------
    # 어떤 종목은 성향과 무관하게 누구나 담는다. 현실에서 그건 대표 지수 ETF와
    # 이름이 널리 알려진 대형주다. 그래서 지수·ETF·중간 위험도에 가산점을 주고,
    # 소수 종목에 쏠리는 현실을 흉내내려고 지프(Zipf) 모양으로 만든다.
    mainstream = (
        3.0 * (items["sector"] == "지수").to_numpy()
        + 1.0 * (items["asset_class"] == "ETF").to_numpy()
        + 1.0 * items["risk_level"].between(2, 4).to_numpy()
        + rng.random(n_items)
    )
    rank = np.empty(n_items)
    rank[np.argsort(-mainstream)] = np.arange(1, n_items + 1)
    global_p = (1.0 / rank) / (1.0 / rank).sum()

    # --- 성향별 선호 분포 ---------------------------------------------------
    persona_p = {p: to_probs(item_scores_for(p, items), PERSONA_SHARPNESS)
                 for p in personas}

    # --- 하위 취향 그룹의 장바구니 -------------------------------------------
    # 같은 성향 안에서도 사람마다 즐겨 담는 종목이 갈린다. 그 갈림을 12개 그룹으로
    # 만들어 둔다.
    #
    # 여기서 두 가지를 지켜야 성향별 인기 추천이 이 그룹들을 흉내내지 못한다.
    #   1) 네 그룹의 장바구니가 서로 겹치지 않아야 한다. 겹치면 성향별 인기가
    #      네 장바구니의 합집합을 그대로 맞혀 버린다.
    #   2) 성향과 무관한 종목을 몇 개씩 섞는다. 실제 사람도 성향에서 벗어난 종목을
    #      몇 개씩 갖고 있다. 이건 성향 정보만으로는 절대 알 수 없고, 함께 담은
    #      사람들을 보고 배우는 행렬분해만 잡아낼 수 있다.
    n_core = SUBCLUSTER_BASKET_SIZE - SUBCLUSTER_QUIRKS
    baskets = {}
    for p in personas:
        pool = rng.permutation(
            np.argsort(-persona_p[p])[: n_core * N_SUBCLUSTERS_PER_PERSONA])
        outside = np.setdiff1d(np.arange(n_items), pool)
        for s in range(N_SUBCLUSTERS_PER_PERSONA):
            core = pool[s * n_core:(s + 1) * n_core]
            quirk = rng.choice(outside, size=SUBCLUSTER_QUIRKS, replace=False)
            picked = np.concatenate([core, quirk])
            w = rng.random(len(picked)) + 0.4
            baskets[(p, s)] = (picked, w / w.sum())

    # --- 투자자 만들기 ------------------------------------------------------
    users, rows = [], []
    end = pd.Timestamp("2026-08-31")
    start = end - pd.DateOffset(months=MONTHS)
    test_start = end - pd.DateOffset(months=TEST_MONTHS)

    for i in range(N_USERS):
        uid = f"U{i+1:04d}"
        persona = personas[i % len(personas)]           # 100명씩 고르게
        sub = int(rng.integers(N_SUBCLUSTERS_PER_PERSONA))

        # 5%는 마지막 달에만 나타나는 신규 투자자다. 3주차에서 "학습 데이터에 없던
        # 사람에게 무엇을 추천할 것인가"(콜드스타트)를 보여주는 데 쓴다.
        newcomer = rng.random() < NEWCOMER_RATE
        joined = (test_start + pd.Timedelta(days=int(rng.integers(0, 25)))) if newcomer \
            else (start + pd.Timedelta(days=int(rng.integers(0, 300))))

        users.append({
            "user_id": uid, "persona": persona, "subcluster": f"{persona}-{sub+1}",
            "risk_appetite": PERSONA_TASTE[persona]["risk_center"],
            "joined_at": joined.date(), "is_newcomer": newcomer,
        })

        lo, hi = INTERACTIONS_PER_USER
        n = int(rng.integers(lo, hi + 1))
        if newcomer:
            n = max(3, n // 4)                          # 신규는 기록이 적다

        basket_items, basket_w = baskets[(persona, sub)]
        span = max((end - joined).days, 1)

        for _ in range(n):
            r = rng.random()
            if r < P_GLOBAL_POPULAR:
                item = rng.choice(n_items, p=global_p)
            elif r < P_GLOBAL_POPULAR + P_PERSONA:
                item = rng.choice(n_items, p=persona_p[persona])
            elif r < P_GLOBAL_POPULAR + P_PERSONA + P_SUBCLUSTER:
                item = rng.choice(basket_items, p=basket_w)
            else:
                item = rng.integers(n_items)

            ts = joined + pd.Timedelta(days=int(rng.integers(0, span)),
                                       hours=int(rng.integers(0, 24)))
            event = rng.choice(["view", "like", "buy"], p=[0.55, 0.25, 0.20])
            rows.append({"user_id": uid, "item_id": items.at[int(item), "item_id"],
                         "ts": ts, "event": event})

    users_df = pd.DataFrame(users)
    inter = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    # 같은 사람이 같은 종목을 여러 번 본 기록은 가장 이른 것만 남긴다.
    inter = inter.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    users_df.to_csv(DATA / "users.csv", index=False)
    inter.to_csv(DATA / "interactions.csv", index=False)

    print(f"users.csv         {len(users_df):>6,}명")
    print(f"interactions.csv  {len(inter):>6,}건  (1인 평균 {len(inter)/len(users_df):.1f}건)")
    print(f"기간              {inter['ts'].min().date()} ~ {inter['ts'].max().date()}")
    print(f"테스트 구간       {test_start.date()} 이후")


if __name__ == "__main__":
    main()
