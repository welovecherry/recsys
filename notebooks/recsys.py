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
    interactions = pd.read_csv(DATA / "interactions.csv", parse_dates=["ts"])
    return items, users, interactions


def split_by_time(interactions, test_months=TEST_MONTHS):
    """시간을 기준으로 나눈다 — 과거로 배우고 미래를 맞힌다.

    3주차에서 다루는 내용이다. 무작위로 나누면 미래의 기록으로 과거를 맞히게 되어
    점수가 실제보다 높게 나온다(데이터 누수). 실제 서비스에서는 미래를 볼 수 없으므로
    시간을 기준으로 나눠야 정직한 점수가 된다.
    """
    cut = interactions["ts"].max() - pd.DateOffset(months=test_months)
    return interactions[interactions["ts"] < cut].copy(), interactions[interactions["ts"] >= cut].copy(), cut


def recall_per_user(recommend, train, test, k=TOP_K):
    """사람마다의 Recall@k 를 {사람: 점수} 로 돌려준다.

    recall_at_k 는 평균 하나만 주는데, 기존·신규처럼 무리를 갈라 보려면 사람별 점수가
    있어야 한다(3주차부터 계속 쓴다). 재는 방식은 recall_at_k 와 한 글자도 다르지 않다 —
    이미 담은 것은 정답에서 빼고, 맞힐 것이 없는 사람은 채점에서 뺀다.

    recommend(user_id, seen) 는 추천할 종목 목록을 돌려주는 함수다.
    """
    seen_map = train.groupby("user_id")["item_id"].apply(set).to_dict()
    scores = {}
    for uid, grp in test.groupby("user_id"):
        seen = seen_map.get(uid, set())
        answer = set(grp["item_id"]) - seen          # 이미 담은 건 정답에서 뺀다
        if not answer:                               # 맞힐 것이 없으면
            continue                                 # 채점에서 뺀다
        got = set(recommend(uid, seen)[:k])
        scores[uid] = len(answer & got) / len(answer)
    return scores


def recall_at_k(recommend, train, test, k=TOP_K):
    """Recall@k — 그 사람이 실제로 담은 것 중 내 추천 10개 안에 몇 개가 들었나.

    recommend(user_id, seen) 는 추천할 종목 목록을 돌려주는 함수다.
    seen 은 그 사람이 학습 구간에서 이미 담은 종목이며, 이미 담은 것을 다시 추천하면
    안 되므로 걸러 내는 데 쓴다.

    반환값은 (평균 점수, 채점한 사람 수) 이다. 사람별 점수가 필요하면 recall_per_user.
    """
    scores = recall_per_user(recommend, train, test, k)
    if not scores:
        return 0.0, 0
    return float(np.mean(list(scores.values()))), len(scores)


def take(ranked, seen, k=TOP_K):
    """인기 순위표에서 아직 안 담은 종목을 위에서부터 k개 고른다."""
    return [i for i in ranked if i not in seen][:k]


def interaction_matrix(train, items):
    """학습 구간을 사람 × 종목 표로 펼친다 — 담았으면 1, 아니면 0.

    4주차에 노트북에서 직접 만들어 본 그 표다. 5주차부터는 표를 만드는 것이
    수업 목표가 아니므로 여기로 옮겼다.
    """
    표 = pd.crosstab(train["user_id"], train["item_id"])
    표 = (표 > 0).astype(int)
    return 표.reindex(columns=items["item_id"], fill_value=0)


def segments(train, items):
    """2주차 세그먼트를 다시 만든다 — 주력 섹터 | 평균 위험도.

    2주차에 노트북에서 손으로 만들어 본 것이다(`레버리지|4` 같은 이름).
    5주차에서는 이것이 후보를 추리는 재료로 쓰일 뿐 수업 목표가 아니라서
    여기로 옮겼다. 만드는 방법이 궁금하면 2주차 노트북을 보면 된다.

    반환값은 ({사람: 세그먼트 이름}, {세그먼트 이름: 많이 담긴 종목 순서}) 이다.
    """
    섹터 = items.set_index("item_id")["sector"].to_dict()
    위험도 = items.set_index("item_id")["risk_level"].to_dict()

    거래 = train.copy()
    거래["섹터"] = 거래["item_id"].map(섹터)
    거래["위험도"] = 거래["item_id"].map(위험도)

    주력섹터 = 거래.groupby("user_id")["섹터"].agg(lambda x: x.value_counts().index[0])
    평균위험도 = 거래.groupby("user_id")["위험도"].mean().round().astype(int)
    이름 = (주력섹터.astype(str) + "|" + 평균위험도.astype(str)).to_dict()

    거래["세그먼트"] = 거래["user_id"].map(이름)
    순위표 = {}
    for 세그, 덩어리 in 거래.groupby("세그먼트"):
        순위표[세그] = list(덩어리["item_id"].value_counts().index)
    return 이름, 순위표


def fill_up(후보, 채울것, n):
    """후보가 n개가 안 되면 뒤를 채운다 — 이미 들어 있는 것은 건너뛴다."""
    후보 = list(후보)
    for 번호 in 채울것:
        if len(후보) >= n:
            break
        if 번호 not in 후보:
            후보.append(번호)
    return 후보


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


def use_korean_font():
    """그래프의 한글이 네모로 나오지 않게, 이 컴퓨터에 깔린 한글 글꼴을 골라 쓴다.

    어느 글꼴이 깔려 있는지는 컴퓨터마다 다르다. 수업에서 가르칠 내용이 아니라
    노트북 밖으로 뺐다. 고른 글꼴 이름을 돌려주고, 하나도 없으면 None 을 돌려준다.
    """
    import matplotlib
    from matplotlib import font_manager

    깔린것 = set()
    for 글꼴 in font_manager.fontManager.ttflist:
        깔린것.add(글꼴.name)

    for 이름 in ["AppleGothic", "Malgun Gothic", "NanumGothic",
                 "NanumBarunGothic", "Noto Sans CJK KR"]:
        if 이름 in 깔린것:
            matplotlib.rcParams["font.family"] = 이름
            matplotlib.rcParams["axes.unicode_minus"] = False   # 마이너스가 네모로 나오는 것을 막는다
            return 이름
    return None


def leaderboard(upto=None):
    """지금까지 쌓인 주차별 점수표를 읽는다.

    upto 를 주면 그 주차까지만 남긴다. **노트북에서는 반드시 그 주차를 넘긴다.**
    작업 저장소에는 뒤 주차의 결과가 이미 쌓여 있어서, 그냥 부르면 학생이 앞으로
    배울 답(행렬분해·후보 생성 …)이 출력에 실려 그대로 배포된다.
    """
    import json
    cols = ["level", "name", "recall_at_10", "note"]
    if not SCORES.exists():
        return pd.DataFrame(columns=cols)
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(SCORES.glob("level_*.json"))]
    if upto is not None:
        rows = [r for r in rows if r["level"] <= int(upto)]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


# ------------------------------------------------------------------ 6주차 FM · MLP
# 6주차 수업 목표는 FM 과 MLP 가 「무엇을 하는지」 이해하고 결과를 비교하는 것이다.
# 학습 데이터를 만드는 과정과 FM 의 학습 계산은 수업 목표가 아니라서 여기로 뺐다.
# FM 은 scikit-learn 에 없어서 numpy 로 직접 짰다 — 설치할 것이 없다.
FEATURE_COLS = ["sector", "asset_class", "risk_level", "theme"]   # 섹터 · 자산군 · 위험도 · 테마


def training_rows(train, items, neg=4, seed=42):
    """학습 데이터를 만든다 — 담은 짝은 1, 담지 않은 짝은 0.

    담은 짝만 있으면 모델은 「무엇이든 1」이라고 배운다. 그래서 사람마다 담지 않은
    종목을 담은 개수의 neg 배만큼 무작위로 골라 0 으로 넣는다 — 네거티브 샘플링.
    seed 를 고정하면 몇 번을 돌려도 같은 종목이 뽑힌다.
    """
    rng = np.random.default_rng(seed)
    전체 = list(items["item_id"])
    줄들 = []
    for 사람, 담은것 in train.groupby("user_id")["item_id"].apply(set).items():
        for 번호 in sorted(담은것):
            줄들.append((사람, 번호, 1))
        안_담은것 = [번호 for 번호 in 전체 if 번호 not in 담은것]
        for 번호 in rng.choice(안_담은것, min(len(안_담은것), neg * len(담은것)), replace=False):
            줄들.append((사람, 번호, 0))
    return pd.DataFrame(줄들, columns=["user_id", "item_id", "label"])


def all_pairs(user_ids, items):
    """점수를 매길 모든 (사용자, 종목) 짝 — 사용자 순서대로, 그 안에서 종목 순서대로."""
    짝 = [(사람, 번호) for 사람 in user_ids for 번호 in items["item_id"]]
    return pd.DataFrame(짝, columns=["user_id", "item_id"])


def one_hot(rows, items, features=True):
    """(사용자, 종목) 짝을 원핫 인코딩한다 — 한 줄에 사용자 1칸 · 종목 1칸 · 상품 정보 4칸이 1.

    features=False 면 상품 정보 없이 사용자 · 종목만 펼친다.
    열 목록을 고정해 두어서, 학습 데이터와 점수 매길 짝의 열이 언제나 같다.
    """
    표 = rows[["user_id", "item_id"]].merge(items[["item_id"] + FEATURE_COLS], on="item_id", how="left")
    열 = ["user_id", "item_id"] + (FEATURE_COLS if features else [])
    범주 = {"user_id": sorted(rows["user_id"].unique())}
    for 이름 in 열[1:]:
        범주[이름] = sorted(items[이름].astype(str).unique())
    for 이름 in 열:
        표[이름] = pd.Categorical(표[이름].astype(str), categories=범주[이름])
    return pd.get_dummies(표[열]).astype(np.float32)


class FM:
    """FM(factorization machine) — 1 인 칸들을 둘씩 짝지어 잠재 요인의 내적을 모두 더한다.

        점수 = b + (1 인 칸의 가중치 합) + (1 인 칸 둘씩 짝지은 내적의 합)
        확률 = 시그모이드(점수)      ← 0 ~ 1 사이로 바꾼다

    사용법은 scikit-learn 모델과 같다 — FM().fit(X, y) 로 학습, predict(X) 로 확률.
    k = 칸마다 붙는 잠재 요인 개수(4주차의 8 과 같은 뜻), epochs = 학습 데이터를 몇 번 돌지.
    """

    def __init__(self, k=8, epochs=80, lr=0.01, reg=1e-4, seed=42):
        self.k, self.epochs, self.lr, self.reg, self.seed = k, epochs, lr, reg, seed

    def _score(self, X):
        XV = X @ self.V
        짝_합 = 0.5 * ((XV ** 2).sum(1) - ((X ** 2) @ (self.V ** 2)).sum(1))   # 모든 짝의 내적 합을 한 번에
        return self.b + X @ self.w + 짝_합

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.columns = list(X.columns)
        X, y = X.values, np.asarray(y, dtype=float)
        self.V = rng.normal(0, 0.05, (X.shape[1], self.k))
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.loss_curve_ = []                              # 에폭마다의 손실 — scikit-learn MLP 와 같은 이름
        for _ in range(self.epochs):                       # 에폭 — 학습 데이터 전체를 한 번 도는 것
            순서 = rng.permutation(len(X))
            for 시작 in range(0, len(X), 256):              # 256줄씩 나눠 조금씩 고친다
                줄 = 순서[시작:시작 + 256]
                xb = X[줄]
                g = 1 / (1 + np.exp(-self._score(xb))) - y[줄]   # 예측 확률 − 정답
                XV = xb @ self.V
                self.b -= self.lr * g.sum()
                self.w -= self.lr * (xb.T @ g + self.reg * self.w)
                self.V -= self.lr * (xb.T @ (g[:, None] * XV)
                                     - self.V * ((xb ** 2).T @ g)[:, None] + self.reg * self.V)
            p = np.clip(1 / (1 + np.exp(-self._score(X))), 1e-7, 1 - 1e-7)
            self.loss_curve_.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))   # 로그 손실
        return self

    def predict(self, X):
        X = X.reindex(columns=self.columns, fill_value=0).values
        return 1 / (1 + np.exp(-self._score(X)))


def demo():
    """이 파일이 제대로 도는지 확인하는 자체 점검."""
    items, users, interactions = load()
    train, test, cut = split_by_time(interactions)
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
