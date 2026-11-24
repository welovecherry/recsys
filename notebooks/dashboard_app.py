"""9주차 산출물 — 7주차 결과 파일을 읽어 투자자별 추천을 화면으로 보여 주는 Streamlit 앱.

    streamlit run dashboard_app.py

화면 세 개 (탭)
  투자자 한 명      투자자를 고르면 그 사람의 추천 10개와 요약
  투자 성향별       안정형 · 성장형 · 공격형 추천의 위험도 분포
  두 투자자 비교    두 사람의 추천에서 같은 종목과 서로 다른 종목

앱은 학습하지 않는다. 7주차 recommend_batch.py 가 만든 결과 파일을 읽어서 보여 주기만 한다.
Streamlit 은 사용자가 상자를 바꿀 때마다 이 파일을 위에서 아래로 다시 실행한다.
"""
from datetime import datetime     # 날짜와 시각을 다루는 도구
from pathlib import Path          # 파일 위치(경로)를 다루는 도구

import pandas as pd               # 표를 다루는 도구
import streamlit as st            # 웹 화면을 만드는 도구

결과_파일 = Path(__file__).resolve().parent / "output" / "recommendations.csv"   # 7주차가 저장한 파일
필요한_열 = ["user_id", "rank", "item_id", "item_name", "risk_level", "persona", "score"]   # 7주차 결과 파일 규격
상한표 = {"안정형": 3, "성장형": 4, "공격형": 5}                                                  # 6주차에서 정한 투자 성향별 위험도 상한

st.set_page_config(page_title="추천 대시보드", layout="wide")
st.title("추천 대시보드")


@st.cache_data
def 결과_읽기():
    """결과 파일을 읽는다. 같은 호출은 저장해 둔 결과를 재사용한다 (캐시)."""
    try:
        return pd.read_csv(결과_파일, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:                                          # 내용이 하나도 없는 파일
        return pd.DataFrame()


# ── 파일이 없으면 이유를 화면에 보여 주고 멈춘다 ──────────────────────────
if not 결과_파일.exists():
    st.error("결과 파일이 없습니다. 7주차의 python recommend_batch.py 를 먼저 실행하세요.")
    st.stop()

추천표 = 결과_읽기()

빠진_열 = [열 for 열 in 필요한_열 if 열 not in 추천표.columns]
if 추천표.empty or 빠진_열:                                                   # 비었거나 규격과 다른 파일이면 이유를 보여 주고 멈춘다
    st.error(f"결과 파일이 비어 있거나 필요한 열이 없습니다: {빠진_열}")
    st.stop()

투자자_목록 = sorted(추천표["user_id"].unique())

# ── 맨 위 요약 · 결과 파일 갱신 시각 ──────────────────────────────────────
갱신_시각 = datetime.fromtimestamp(결과_파일.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
요약1, 요약2, 요약3, 요약4 = st.columns(4)
요약1.metric("투자자 수", f"{len(투자자_목록)}명")
요약2.metric("추천 줄 수", f"{len(추천표):,}줄")
요약3.metric("결과 파일 갱신", 갱신_시각)
if 요약4.button("캐시 지우기"):                                               # 결과 파일이 바뀌었을 때 옛 결과를 버린다
    st.cache_data.clear()
    st.rerun()

탭_한명, 탭_성향, 탭_비교 = st.tabs(["투자자 한 명", "투자 성향별", "두 투자자 비교"])


# ── 화면 1. 투자자 한 명 ─────────────────────────────────────────────────
with 탭_한명:
    투자자 = st.selectbox("투자자를 고르세요", 투자자_목록)                  # 고르는 상자
    그_사람 = 추천표[추천표["user_id"] == 투자자].sort_values("rank")        # 그 투자자의 줄만, 순위 순서

    성향 = 그_사람["persona"].iloc[0]
    최고_위험도 = int(그_사람["risk_level"].max())
    평균_점수 = float(그_사람["score"].mean())

    왼쪽, 가운데, 오른쪽 = st.columns(3)                                     # 숫자 세 개를 나란히 둔다
    왼쪽.metric("투자 성향", 성향)
    가운데.metric("추천 중 가장 높은 위험도", 최고_위험도)
    오른쪽.metric("추천 점수 평균", f"{평균_점수:.4f}")

    상한 = 상한표[성향]                                                      # 이 투자 성향의 위험도 상한
    if 최고_위험도 <= 상한:
        st.success(f"위험도 상한 {상한} 이 지켜졌습니다 — 추천 10개의 가장 높은 위험도는 {최고_위험도} 입니다.")
    else:
        st.error(f"위험도 상한 {상한} 을 넘은 추천이 있습니다 — 가장 높은 위험도는 {최고_위험도} 입니다.")

    표 = 그_사람[["rank", "item_id", "item_name", "risk_level", "score"]]
    st.dataframe(표, hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=float(추천표["score"].max()), format="%.4f")})   # 점수를 막대로 보여 준다
    st.download_button("추천 10개를 CSV 로 내려받기", 표.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{투자자}_추천.csv", mime="text/csv")


# ── 화면 2. 투자 성향별 ──────────────────────────────────────────────────
with 탭_성향:
    분포 = pd.crosstab(추천표["persona"], 추천표["risk_level"])               # 성향 × 위험도 개수 표
    분포 = 분포.reindex(["안정형", "성장형", "공격형"], fill_value=0)           # 성향 순서를 노트북 9.3 과 같게 한다
    st.write("투자 성향별로 추천된 종목의 위험도 개수")
    st.bar_chart(분포, sort=False)                                          # sort=False : 위 순서 그대로 막대를 그린다
    st.dataframe(분포)

    추천표_상한 = 추천표.copy()
    추천표_상한["상한"] = 추천표_상한["persona"].map(상한표)                    # 줄마다 그 성향의 상한을 붙인다
    추천표_상한["초과"] = 추천표_상한["risk_level"] > 추천표_상한["상한"]         # 상한을 넘으면 True
    점검표 = pd.DataFrame({"위험도 상한": pd.Series(상한표),
                          "추천 수": 추천표_상한.groupby("persona").size(),
                          "상한 초과 추천 수": 추천표_상한.groupby("persona")["초과"].sum()}).reindex(["안정형", "성장형", "공격형"])
    st.write("위험도 상한 점검")
    st.dataframe(점검표)
    if int(점검표["상한 초과 추천 수"].sum()) == 0:
        st.success("세 성향 모두 위험도 상한을 넘은 추천이 0개입니다.")
    else:
        st.error("위험도 상한을 넘은 추천이 있습니다. 위 표에서 성향을 확인하세요.")


# ── 화면 3. 두 투자자 비교 ───────────────────────────────────────────────
with 탭_비교:
    열A, 열B = st.columns(2)
    투자자_A = 열A.selectbox("투자자 A", 투자자_목록, index=0)
    투자자_B = 열B.selectbox("투자자 B", 투자자_목록, index=min(2, len(투자자_목록) - 1))

    if 투자자_A == 투자자_B:
        st.warning("서로 다른 투자자를 고르세요.")
    else:
        줄_A = 추천표[추천표["user_id"] == 투자자_A]
        줄_B = 추천표[추천표["user_id"] == 투자자_B]
        종목_A = set(줄_A["item_id"])
        종목_B = set(줄_B["item_id"])
        같은_종목 = sorted(종목_A & 종목_B)                                   # & = 두 사람 모두에게 추천된 종목
        A에만 = sorted(종목_A - 종목_B)                                       # - = 한쪽에만 있는 종목
        B에만 = sorted(종목_B - 종목_A)

        st.write(f"{투자자_A} ({줄_A['persona'].iloc[0]}) 와 {투자자_B} ({줄_B['persona'].iloc[0]}) 에게 모두 추천된 종목 {len(같은_종목)}개 / 10개")
        st.write(f"평균 위험도 — {투자자_A} {줄_A['risk_level'].mean():.2f} · {투자자_B} {줄_B['risk_level'].mean():.2f}")

        종목정보 = 추천표.drop_duplicates("item_id").set_index("item_id")[["item_name", "risk_level"]]   # 종목 번호 → 이름 · 위험도
        왼쪽, 가운데, 오른쪽 = st.columns(3)
        for 해당_열, 제목, 번호들 in [(왼쪽, f"{투자자_A} 에게만 ({len(A에만)}개)", A에만),
                                      (가운데, f"둘 다 ({len(같은_종목)}개)", 같은_종목),
                                      (오른쪽, f"{투자자_B} 에게만 ({len(B에만)}개)", B에만)]:
            해당_열.write(제목)
            if 번호들:
                해당_열.dataframe(종목정보.loc[번호들], hide_index=True, width="stretch")   # 종목 이름과 위험도만 보여 준다
