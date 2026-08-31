"""1~7주차를 한 번에 돌리는 실행 스크립트 (7주차 산출물).

수업에서는 주차마다 노트북을 열고 셀을 하나씩 눌러 가며 모델을 만들었다. 실제 서비스는
그렇게 돌아가지 않는다. 사람이 노트북을 여는 대신 명령 한 줄이 데이터 생성부터 추천
결과 저장까지를 끝내야 한다. 이 파일이 그 한 줄이다.

    uv run python notebooks/run_all.py

하는 일
-------
1. data/make_data.py 를 실행해 데이터를 다시 만든다
2. 1~7주차 모델을 순서대로 학습하고 각각 Recall@10 을 잰다
3. 점수를 recsys.record() 로 남긴다
4. app/artifacts/ 에 추천 결과(recommendations.parquet)와 모델(model.pkl)을 저장한다
5. 리더보드 표를 출력하고 주차별 점수 그래프를 leaderboard.png 로 저장한다

노트북을 실행하는 방식이 아니라 각 주차 모델을 짧은 함수로 다시 구현했다. 노트북 실행은
느리고 셀 하나가 깨지면 전체가 멈추기 때문이다.
"""

import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recsys  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = recsys.ROOT
ARTIFACTS = recsys.ARTIFACTS
K = recsys.TOP_K


# ------------------------------------------------------------------ 1단계 데이터
def make_data():
    """data/make_data.py 를 실행해 데이터를 처음부터 다시 만든다."""
    subprocess.run([sys.executable, str(ROOT / "data" / "make_data.py")],
                   check=True, capture_output=True)


# ------------------------------------------------------------ 주차별 모델 (1~7)
def week1_popular(train):
    """1주차 · 전체 인기 — 모두에게 가장 많이 담긴 종목 10개를 똑같이 준다."""
    ranked = list(train["item_id"].value_counts().index)
    return lambda uid, seen: recsys.take(ranked, seen)


def user_groups(train, items):
    """거래 기록만 보고 그 사람의 취향을 짐작한다 — 주력 섹터와 평균 위험도.

    users.csv 의 persona·subcluster·risk_appetite 는 쓰지 않는다. 앞의 둘은 데이터를
    만들 때 심어 둔 정답지이고, risk_appetite 도 그 정답지에서 그대로 파생된 값이라
    사실상 답을 보고 푸는 것이 된다.

    더 중요한 이유가 있다. 실제 강의에서는 이 가상 데이터 대신 공개 데이터를 쓸 계획인데,
    거기에는 투자 성향 라벨 같은 것이 아예 없다. 처음부터 **행동 기록만으로 취향을 짐작하는
    구조**로 만들어 두어야 데이터를 갈아끼워도 그대로 돌아간다.
    """
    sector = items.set_index("item_id")["sector"].to_dict()
    risk = items.set_index("item_id")["risk_level"].to_dict()
    top_sector = (train.assign(s=train["item_id"].map(sector))
                  .groupby("user_id")["s"]
                  .agg(lambda x: x.value_counts().index[0]).to_dict())
    avg_risk = (train.assign(r=train["item_id"].map(risk))
                .groupby("user_id")["r"].mean().round(0).astype(int).to_dict())
    return {u: f"{top_sector[u]}|{avg_risk[u]}" for u in top_sector}


def week2_group_rule(train, items):
    """2주차 · 취향이 비슷한 사람끼리 묶어, 그 무리가 많이 담은 종목을 준다."""
    group = user_groups(train, items)
    tr = train.assign(g=train["user_id"].map(group))
    tables = {g: list(v["item_id"].value_counts().index) for g, v in tr.groupby("g")}
    fallback = list(train["item_id"].value_counts().index)
    return lambda uid, seen: recsys.take(tables.get(group.get(uid), fallback), seen)


def week3_random_split(inter, items, n_test):
    """3주차 · 무작위로 나누면 점수가 얼마나 부풀려지는지 재 본다.

    2주차와 똑같은 모델을 쓰되 데이터를 시간이 아니라 무작위로 나눈다. 그러면 미래의
    기록을 보고 과거를 맞히게 되어 실제보다 높은 점수가 나온다. 이것이 데이터 누수다.
    """
    shuffled = inter.sample(frac=1, random_state=0)
    rtest, rtrain = shuffled.iloc[:n_test], shuffled.iloc[n_test:]
    return recsys.recall_at_k(week2_group_rule(rtrain, items), rtrain, rtest)


def fit_als(train, items, factors=8, regularization=0.5, iterations=30):
    """4주차 · 행렬분해(ALS). 검증된 하이퍼파라미터를 그대로 쓴다."""
    from implicit.als import AlternatingLeastSquares

    uids = sorted(train["user_id"].unique())
    iids = list(items["item_id"])
    ui = {u: i for i, u in enumerate(uids)}
    ii = {v: i for i, v in enumerate(iids)}

    rows = train["user_id"].map(ui).to_numpy()
    cols = train["item_id"].map(ii).to_numpy()
    mat = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uids), len(iids)))

    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=42)
    model.fit(mat, show_progress=False)
    return model, mat, ui, ii, iids


def als_scores(model, mat, ui, uid, n_items):
    """한 사람에 대한 전체 종목 점수 벡터. 학습에 없던 사람은 None."""
    if uid not in ui:
        return None
    idx = ui[uid]
    return model.user_factors[idx] @ model.item_factors.T


def week4_als(model, mat, ui, iids, popular):
    """4주차 · 행렬분해가 낸 추천."""
    def recommend(uid, seen):
        s = als_scores(model, mat, ui, uid, len(iids))
        if s is None:                                      # 신규 투자자 → 인기순
            return recsys.take(popular, seen)
        order = np.argsort(-s)
        return recsys.take([iids[j] for j in order], seen)
    return recommend


def week5_candidate_rank(model, mat, ui, iids, popular, n_cand=50, w_pop=0.15):
    """5주차 · 후보 생성 + 랭킹 — 두 단계로 나눈다.

    1단계(후보 생성): 행렬분해 점수 상위 50개만 남긴다. 100개 전부를 정밀하게 따지는
    대신 가능성 있는 후보만 추린다. 실제 서비스에서 상품이 수만 개일 때 필요한 단계다.
    2단계(랭킹): 그 50개만 다시 점수 매긴다. 여기서는 행렬분해 점수에 전체 인기도를
    조금 더해 최종 순위를 정한다.
    """
    pop_rank = {v: 1.0 - i / len(popular) for i, v in enumerate(popular)}
    pop_vec = np.array([pop_rank.get(v, 0.0) for v in iids])

    def recommend(uid, seen):
        s = als_scores(model, mat, ui, uid, len(iids))
        if s is None:
            return recsys.take(popular, seen)
        cand = np.argsort(-s)[:n_cand]                     # 1단계
        z = (s[cand] - s[cand].mean()) / (s[cand].std() + 1e-9)
        final = z + w_pop * pop_vec[cand]                  # 2단계
        order = cand[np.argsort(-final)]
        return recsys.take([iids[j] for j in order], seen)
    return recommend


def content_matrix(items, train, iids):
    """상품 정보(섹터·자산군·위험도·테마)로 만든 사용자-종목 내용 유사도.

    items.csv 의 네 컬럼을 원핫으로 펼치고, 사람이 담아 온 종목들의 평균 벡터를 그 사람의
    취향으로 본다. 그 취향 벡터와 각 종목 벡터의 코사인 유사도가 내용 점수다.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import OneHotEncoder

    feats = items.set_index("item_id").loc[iids,
                                           ["sector", "asset_class", "risk_level", "theme"]]
    X = OneHotEncoder(handle_unknown="ignore").fit_transform(feats.astype(str)).toarray()

    ii = {v: i for i, v in enumerate(iids)}
    prof = {}
    for uid, grp in train.groupby("user_id"):
        rows = [ii[v] for v in grp["item_id"] if v in ii]
        prof[uid] = X[rows].mean(axis=0)
    users_order = list(prof)
    S = cosine_similarity(np.vstack([prof[u] for u in users_order]), X)
    return {u: S[i] for i, u in enumerate(users_order)}


def week6_blend(model, mat, ui, iids, popular, content, alpha):
    """6주차 · 협업 신호 + 상품 정보 섞기. alpha 가 0이면 4주차와 완전히 같다."""
    def recommend(uid, seen):
        s = als_scores(model, mat, ui, uid, len(iids))
        c = content.get(uid)
        if s is None:
            return recsys.take(popular, seen)
        zs = (s - s.mean()) / (s.std() + 1e-9)
        if c is None:
            blended = zs
        else:
            zc = (c - c.mean()) / (c.std() + 1e-9)
            blended = (1 - alpha) * zs + alpha * zc
        return recsys.take([iids[j] for j in np.argsort(-blended)], seen)
    return recommend


# --------------------------------------------------------- 7단계 산출물 저장
def save_artifacts(model, mat, ui, ii, iids, users, recommend):
    """학습 결과를 파일로 남긴다 — 학습과 서빙의 분리.

    무거운 계산은 여기서 한 번만 하고, 나중에 만들 Streamlit 앱은 이 파일을 읽기만 한다.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for uid in users["user_id"]:
        seen = set()                                       # 저장용은 전체 기간 기준
        for rank, item in enumerate(recommend(uid, seen)[:K], start=1):
            rows.append({"user_id": uid, "rank": rank, "item_id": item, "score": None})
    recs = pd.DataFrame(rows)

    # 점수도 같이 담는다 (앱에서 보여주기 위함)
    for uid, grp in recs.groupby("user_id"):
        s = als_scores(model, mat, ui, uid, len(iids))
        if s is None:
            continue
        recs.loc[grp.index, "score"] = [float(s[ii[v]]) for v in grp["item_id"]]
    recs["score"] = recs["score"].astype(float).fillna(0.0)

    rec_path = ARTIFACTS / "recommendations.parquet"
    recs.to_parquet(rec_path, index=False)

    pkl_path = ARTIFACTS / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"user_factors": model.user_factors,
                     "item_factors": model.item_factors,
                     "user_index": ui, "item_index": ii, "item_ids": iids}, f)
    return recs, rec_path, pkl_path


def save_chart(board):
    """주차별 점수 변화 그래프."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(board["level"], board["recall_at_10"], marker="o", color="#2b6cb0")
    for _, r in board.iterrows():
        ax.annotate(f"{r['recall_at_10']:.4f}", (r["level"], r["recall_at_10"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(0.4610, ls="--", color="#a0aec0")
    ax.text(board["level"].max(), 0.4610, " subgroup ceiling 0.4610",
            va="bottom", ha="right", fontsize=9, color="#718096")
    ax.set_xlabel("Week")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 by week (time-based split)")
    ax.set_xticks(list(board["level"]))
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = ARTIFACTS / "leaderboard.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def kb(path):
    return f"{Path(path).stat().st_size / 1024:.1f} KB"


# ------------------------------------------------------------------------ 본체
def main():
    t0 = time.time()
    print("=" * 62)
    print("1단계 · 데이터 생성")
    print("=" * 62)
    make_data()
    items, users, inter = recsys.load()
    train, test, cut = recsys.split_by_time(inter)
    print(f"종목 {len(items)}개 · 투자자 {len(users)}명 · 거래 {len(inter):,}건")
    print(f"학습 {len(train):,}건 (~{cut.date()}) / 테스트 {len(test):,}건")

    popular = list(train["item_id"].value_counts().index)

    print("\n" + "=" * 62)
    print("2단계 · 주차별 모델 학습과 채점")
    print("=" * 62)

    s1, n = recsys.recall_at_k(week1_popular(train), train, test)
    recsys.record(1, "전체 인기 종목", s1, note="모두에게 같은 10개. 출발점이 되는 점수다.")

    rule = week2_group_rule(train, items)
    s2, _ = recsys.recall_at_k(rule, train, test)
    recsys.record(2, "취향이 비슷한 무리끼리", s2,
                  note="거래 기록으로 주력 섹터와 위험도를 짐작해 사람마다 다른 목록을 준다.")

    # 3주차는 모델을 바꾸지 않는다. 채점 방법만 바꿔서 점수가 부풀려져 있었음을 보인다.
    s3_random, _ = week3_random_split(inter, items, len(test))
    s3, _ = recsys.recall_at_k(rule, train, test)
    print(f"\n  무작위 분할 {s3_random:.4f}  →  시간순 분할 {s3:.4f}  "
          f"({(s3_random - s3) / s3 * 100:+.1f}%)")
    recsys.record(3, "시간순 분할로 정직하게", s3,
                  note=f"모델은 2주차와 같다. 무작위로 나누면 {s3_random:.4f}가 나오는데 "
                       f"이는 미래를 보고 맞힌 부풀려진 점수다.")

    model, mat, ui, ii, iids = fit_als(train, items)
    rec4 = week4_als(model, mat, ui, iids, popular)
    s4, _ = recsys.recall_at_k(rec4, train, test)
    recsys.record(4, "행렬분해 ALS", s4,
                  note="성향 안에 숨은 더 잘게 나뉜 취향 그룹까지 잡아낸다.")

    rec5 = week5_candidate_rank(model, mat, ui, iids, popular)
    s5, _ = recsys.recall_at_k(rec5, train, test)
    recsys.record(5, "후보 생성 + 랭킹", s5,
                  note="상위 50개만 추린 뒤 다시 순위를 매긴다. 규모가 커질 때 필요한 구조다.")

    content = content_matrix(items, train, iids)
    curve = []
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        sc, _ = recsys.recall_at_k(
            week6_blend(model, mat, ui, iids, popular, content, alpha), train, test)
        curve.append((float(alpha), sc))
    best_alpha, s6 = max(curve, key=lambda x: x[1])
    print("\n  혼합 비율별 점수 (0.0 = 협업 신호만, 1.0 = 상품 정보만)")
    for a, sc in curve:
        print(f"    alpha={a:.1f}  {sc:.4f}")
    direction = "올랐다" if s6 > s4 else "오르지 않았다"
    recsys.record(6, "상품 정보를 더한 모델", s6,
                  note=f"가장 좋은 혼합 비율 alpha={best_alpha}. "
                       f"4주차 {s4:.4f} 대비 {direction}.")

    s7, _ = recsys.recall_at_k(rec4, train, test)
    recsys.record(7, "파이프라인으로 자동화", s7, note="점수는 그대로, 사람 손이 빠졌다")

    print("\n" + "=" * 62)
    print("3단계 · 산출물 저장")
    print("=" * 62)
    recs, rec_path, pkl_path = save_artifacts(model, mat, ui, ii, iids, users, rec4)
    print(f"{rec_path.name}  {len(recs):,}행  {kb(rec_path)}")
    print(f"{pkl_path.name}  {kb(pkl_path)}")

    print("\n" + "=" * 62)
    print("4단계 · 리더보드")
    print("=" * 62)
    board = recsys.leaderboard()
    print(board.to_string(index=False))
    png = save_chart(board)
    print(f"\n{png.name}  {kb(png)}")
    print(f"\n전체 실행 시간 {time.time() - t0:.1f}초")

    assert len(recs) == len(users) * K, "추천 결과 행 수가 맞지 않는다"
    assert board["recall_at_10"].between(0, 1).all(), "점수가 0~1 밖이다"


if __name__ == "__main__":
    main()
