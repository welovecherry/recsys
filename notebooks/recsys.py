"""모든 주차가 함께 쓰는 도구 상자.

주차마다 모델은 달라지지만 **데이터를 나누는 방법과 점수를 재는 방법은 끝까지 같아야
한다.** 그래야 1주차 점수와 6주차 점수를 나란히 비교할 수 있다. 주차마다 채점 방식이
조금씩 달라지면 리더보드의 숫자는 아무 의미가 없어진다.

그래서 그 두 가지를 여기에 한 번만 적어 두고 모든 노트북이 가져다 쓴다.

주의 — 2주차에는 이 파일의 recall_at_k 를 쓰지 않는다. 2주차 수업 목표가 "평가 지표를
직접 계산해 보는 것"이므로, 그때는 노트북 안에서 손으로 짜 보고 이 함수와 같은 값이
나오는지 맞춰 본다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "app" / "artifacts"

TOP_K = 10          # 추천 목록의 길이. 10주 내내 바뀌지 않는다.
TEST_MONTHS = 1     # 마지막 1개월이 채점용 정답 구간이다.


def load():
    """종목·투자자·거래 기록을 읽는다."""
    items = pd.read_csv(DATA / "items.csv")
    users = pd.read_csv(DATA / "users.csv")
    inter = pd.read_csv(DATA / "interactions.csv", parse_dates=["ts"])
    return items, users, inter


def split_by_time(inter, test_months=TEST_MONTHS):
    """시간을 기준으로 나눈다 — 과거로 배우고 미래를 맞힌다.

    3주차에서 다루는 내용이다. 무작위로 나누면 미래의 기록으로 과거를 맞히게 되어
    점수가 실제보다 높게 나온다(데이터 누수). 실제 서비스에서는 미래를 볼 수 없으므로
    시간을 기준으로 나눠야 정직한 점수가 된다.
    """
    cut = inter["ts"].max() - pd.DateOffset(months=test_months)
    return inter[inter["ts"] < cut].copy(), inter[inter["ts"] >= cut].copy(), cut


def recall_at_k(recommend, train, test, k=TOP_K):
    """Recall@k — 그 사람이 실제로 담은 것 중 내 추천 10개 안에 몇 개가 들었나.

    recommend(user_id, seen) 는 추천할 종목 목록을 돌려주는 함수다.
    seen 은 그 사람이 학습 구간에서 이미 담은 종목이며, 이미 담은 것을 다시 추천하면
    안 되므로 걸러 내는 데 쓴다.

    반환값은 (평균 점수, 채점한 사람 수) 이다.
    """
    seen_map = train.groupby("user_id")["item_id"].apply(set).to_dict()
    scores = []
    for uid, grp in test.groupby("user_id"):
        seen = seen_map.get(uid, set())
        answer = set(grp["item_id"]) - seen          # 이미 담은 건 정답에서 뺀다
        if not answer:
            continue
        got = set(recommend(uid, seen)[:k])
        scores.append(len(answer & got) / len(answer))
    return float(np.mean(scores)), len(scores)


def take(ranked, seen, k=TOP_K):
    """인기 순위표에서 아직 안 담은 종목을 위에서부터 k개 고른다."""
    return [i for i in ranked if i not in seen][:k]


# ------------------------------------------------------------------ 리더보드
# 주차마다 파일을 따로 쓴다. 한 파일에 모아 쓰면 여러 노트북을 동시에 돌릴 때
# 서로 덮어써서 점수가 사라진다.
SCORES = ARTIFACTS / "scores"


def record(level, name, score, note=""):
    """주차별 점수를 남긴다. 같은 주차를 다시 돌리면 덮어쓴다.

    점수가 떨어지는 주차도 그대로 기록한다. 왜 떨어졌는지 설명하는 것이 그 주차의
    학습 목표이기 때문이다.
    """
    import json
    SCORES.mkdir(parents=True, exist_ok=True)
    row = {"level": int(level), "name": name,
           "recall_at_10": round(float(score), 4), "note": note}
    (SCORES / f"level_{int(level):02d}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"레벨 {level} · {name} · Recall@10 = {score:.4f}")
    return row


def leaderboard():
    """지금까지 쌓인 주차별 점수표를 읽는다."""
    import json
    cols = ["level", "name", "recall_at_10", "note"]
    if not SCORES.exists():
        return pd.DataFrame(columns=cols)
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(SCORES.glob("level_*.json"))]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


def demo():
    """이 파일이 제대로 도는지 확인하는 자체 점검."""
    items, users, inter = load()
    train, test, cut = split_by_time(inter)
    assert len(train) > 0 and len(test) > 0, "분할 결과가 비어 있다"
    assert train["ts"].max() < test["ts"].min(), "학습 구간이 테스트 구간보다 뒤에 있다"

    popular = list(train["item_id"].value_counts().index)
    score, n = recall_at_k(lambda u, seen: take(popular, seen), train, test)
    assert 0.0 <= score <= 1.0, "점수가 0~1 범위를 벗어났다"
    assert n > 0, "채점된 사람이 없다"

    # 아무 종목이나 10개 고르면 인기순보다 나빠야 정상이다.
    rand = list(items["item_id"].sample(frac=1, random_state=0))
    rand_score, _ = recall_at_k(lambda u, seen: take(rand, seen), train, test)
    assert score > rand_score, "인기순이 무작위보다 못하다 — 데이터가 이상하다"

    print(f"자체 점검 통과 · 인기순 {score:.4f} > 무작위 {rand_score:.4f} · {n}명 채점")


if __name__ == "__main__":
    demo()
