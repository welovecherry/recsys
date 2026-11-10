"""7주차 산출물 — 실행 한 번으로 전원의 추천 결과 파일을 만든다.

    python recommend_batch.py

하는 일 (6주차까지 노트북에서 셀을 하나씩 실행하던 일을 하나의 파일로 만들었다) — 파이프라인(pipeline)
  ① 들어온 데이터에 빈 값 · 중복 · 없는 번호가 없는지 검사한다   — 데이터 점검
  ② 상호작용 기록 전체로 모델을 학습한다                            — 학습(training)
  ③ 300명 모두에게 위험도 상한을 지킨 추천 10개                 — 일괄 추천(batch recommendation)
  ④ 결과 표가 정해 둔 규격을 지켰는지 검사한다                  — 규격 검사
  ⑤ 통과하면 CSV 파일 하나에 저장한다                           — 결과 파일
  ① 이나 ④ 에서 문제가 나오면 그 단계에서 실행을 중단한다. 잘못된 파일은 저장하지 않는다.

다음 주 API(FastAPI)는 이 파일을 읽기만 한다 — 서빙(serving). API 는 모델을 다시 학습하지 않는다.
"""
from pathlib import Path          # 파일 위치(경로)를 다루는 도구

import numpy as np                # 숫자 계산 도구 (점수 정렬에 쓴다)
import pandas as pd               # 표를 다루는 도구
from sklearn.decomposition import TruncatedSVD   # 행렬분해 모델 (4주차부터 쓰던 것)

import recsys                     # 이 수업용 도구 모음 — 데이터 읽기 · 사용자-아이템 행렬 · 이미 담은 종목 빼기

# ── 설정값 — 스크립트 전체에서 같이 쓰는 값은 맨 위에 한 번만 적는다 ──
출력_파일 = Path("output") / "recommendations.csv"       # 결과 파일을 저장할 위치
상한 = {"안정형": 3, "성장형": 4, "공격형": 5}          # 6주차 적합 기준 — 투자성향별 최대 위험도
규격_열 = ["user_id", "rank", "item_id", "item_name", "risk_level", "persona", "score"]   # 결과 파일의 열 이름과 순서 (output schema)


def 데이터_점검(items, users, 기록):
    """들어온 데이터를 검사해 {검사 이름: 문제 개수} 를 돌려준다. 모두 0 이면 통과."""
    # 검사마다 「문제가 있는 줄」을 True 로 표시하고(isna · duplicated · isin · between), sum() 으로 개수를 센다
    # `~` 는 True 와 False 를 뒤집는다 — isin 은 「있는가」이므로 앞에 붙여서 「없는 것」을 센다
    return {"빈 값": int(items.isna().sum().sum() + users.isna().sum().sum() + 기록.isna().sum().sum()),   # 표 세 개의 비어 있는 값을 모두 센다
            "똑같은 상호작용 기록": int(기록.duplicated().sum()),                                         # 모든 열이 같은 줄이 앞에 이미 나온 경우
            "종목 목록에 없는 종목 번호": int((~기록["item_id"].isin(items["item_id"])).sum()),             # 종목 표에 없는 번호
            "투자자 목록에 없는 투자자 번호": int((~기록["user_id"].isin(users["user_id"])).sum()),         # 투자자 표에 없는 번호
            "세 가지가 아닌 투자성향": int((~users["persona"].isin(상한)).sum()),                          # 딕셔너리 `상한` 의 키(안정형 · 성장형 · 공격형)와 비교한다
            "1 ~ 5 를 벗어난 위험도": int((~items["risk_level"].between(1, 5)).sum())}                    # between(1, 5) = 1 이상 5 이하


def 학습하기(기록, items):
    """상호작용 기록으로 TruncatedSVD 를 학습하고, 서빙에 필요한 것을 딕셔너리 하나에 담는다."""
    상호작용_행렬 = recsys.interaction_matrix(기록, items)   # 사용자 × 종목 표. 상호작용이 있으면 1, 없으면 0
    모델 = TruncatedSVD(n_components=8, random_state=42)    # random_state 고정 = 다시 실행해도 같은 결과
    return {"사용자_요인": 모델.fit_transform(상호작용_행렬),   # 사용자 × 8 크기의 행렬 (n_components=8)
            "아이템_요인": 모델.components_,                  # 8 × 종목 크기의 행렬. 사용자 요인과 곱하면 점수가 된다
            "사용자_목록": list(상호작용_행렬.index),          # 사용자_요인의 몇 번째 줄이 누구인지 알려 주는 투자자 번호 목록
            "아이템_목록": list(상호작용_행렬.columns),        # 아이템_요인의 몇 번째 열이 어느 종목인지 알려 주는 종목 번호 목록
            "인기순위": list(기록["item_id"].value_counts().index),   # 상호작용이 많은 종목 순서 — 기록이 없는 투자자에게 쓴다
            "이미_담은것": 기록.groupby("user_id")["item_id"].apply(set).to_dict()}   # {투자자: 이미 상호작용한 종목 번호 모음} — 추천에서 뺀다


def 추천하기(묶음, 사람, 성향, 위험도):
    """한 사람에게 위험도 상한을 지킨 추천 10개를 (종목 번호, 점수) 목록으로 준다."""
    점수_딕셔너리 = {}                                       # {종목 번호: 점수}. 기록이 없는 사람은 비어 있다
    if 사람 in 묶음["사용자_목록"]:                           # 학습에 쓴 기록이 있는 사람
        점수 = 묶음["사용자_요인"][묶음["사용자_목록"].index(사람)] @ 묶음["아이템_요인"]   # @ = 행렬 곱. 종목 100개의 점수가 한 번에 나온다
        순서 = []
        for 자리 in np.argsort(-점수):                        # 점수가 높은 순서대로 종목의 위치를 꺼낸다 (- 를 붙이면 높은 것이 먼저)
            순서.append(묶음["아이템_목록"][자리])               # 그 위치의 종목 번호
            점수_딕셔너리[묶음["아이템_목록"][자리]] = float(점수[자리])
    else:
        순서 = 묶음["인기순위"]                              # 기록이 없는 사람은 인기 순서
    통과 = []
    for 번호 in 순서:                                       # 순서를 지키면서 위험도 상한을 넘는 종목을 거른다
        if 위험도[번호] <= 상한[성향[사람]]:                  # 상한[성향[사람]] = 이 사람의 투자 성향이 허용하는 최대 위험도
            통과.append(번호)
    결과 = []
    for 번호 in recsys.take(통과, 묶음["이미_담은것"].get(사람, set())):   # 이미 담은 종목을 빼고 위에서 10개 (recsys.take)
        결과.append((번호, round(점수_딕셔너리.get(번호, 0.0), 4)))      # 점수는 소수 넷째 자리까지. 기록이 없는 사람은 0
    return 결과


def 일괄_추천(묶음, users, items):
    """300명 모두의 추천을 규격대로 표 하나에 담는다."""
    종목 = items.set_index("item_id")                        # item_id 를 인덱스(행을 찾는 기준)로 바꿔서 종목 번호로 바로 찾는다
    성향 = users.set_index("user_id")["persona"].to_dict()   # {투자자 번호: 투자 성향}
    위험도 = 종목["risk_level"].to_dict()                    # {종목 번호: 위험도}
    줄들 = []                                               # 결과 파일의 한 줄 = 한 투자자의 추천 한 개
    for 사람 in users["user_id"]:
        순위 = 1                                            # 1 = 가장 먼저 추천하는 종목
        for 번호, 점수 in 추천하기(묶음, 사람, 성향, 위험도):   # 이미 점수가 높은 순서로 나온다
            줄들.append({"user_id": 사람, "rank": 순위, "item_id": 번호,
                         "item_name": 종목.loc[번호, "name"], "risk_level": int(위험도[번호]),
                         "persona": 성향[사람], "score": 점수})
            순위 = 순위 + 1
    return pd.DataFrame(줄들, columns=규격_열)               # columns 로 열 이름과 순서를 규격에 맞춘다


def 규격_검사(표, users, 묶음):
    """결과 표가 규격을 지켰는지 검사하고, 어긴 것을 목록으로 돌려준다. 빈 목록이면 통과."""
    문제 = []                                               # 어긴 것을 하나씩 모은다. 끝까지 비어 있으면 통과
    if list(표.columns) != 규격_열:                          # ① 열 이름과 순서
        문제.append("열 이름이나 순서가 규격과 다릅니다")
    if 표["user_id"].nunique() != len(users):                # ② 투자자 300명 모두에게 추천이 있는가 (nunique = 서로 다른 값의 개수)
        문제.append("추천이 없는 사람이 있습니다")
    if (표.groupby("user_id").size() != 10).any():           # ③ 투자자마다 줄 수가 정확히 10인가 (any = 하나라도 해당하면 True)
        문제.append("추천이 10개가 아닌 사람이 있습니다")
    for _, 줄 in 표.iterrows():                              # 아래 두 검사는 줄을 하나씩 확인한다
        if 줄["risk_level"] > 상한[줄["persona"]]:            # ④ 투자 성향의 위험도 상한을 넘는가
            문제.append(f"{줄['user_id']} 에게 상한을 넘는 {줄['item_id']}")
        if 줄["item_id"] in 묶음["이미_담은것"].get(줄["user_id"], set()):   # ⑤ 이미 상호작용한 종목을 추천했는가
            문제.append(f"{줄['user_id']} 에게 이미 담은 {줄['item_id']}")
    return 문제


def main():
    items, users, interactions = recsys.load()               # data 폴더의 CSV 세 개를 표로 읽는다

    # ① 데이터 점검 — 하나라도 문제가 있으면 여기서 멈춘다. 학습도 저장도 하지 않는다
    점검 = 데이터_점검(items, users, interactions)
    if sum(점검.values()) > 0:
        raise SystemExit(f"① 데이터 점검 실패 — {점검}")      # SystemExit = 스크립트를 끝낸다. 메시지가 화면에 나온다
    print(f"① 데이터 점검 — 여섯 가지 모두 0건 · 통과")

    # ② 학습 — 시간순으로 나누지 않고 전체를 쓴다. 채점할 미래가 없고 가장 최근 기록까지 반영해야 하기 때문이다
    묶음 = 학습하기(interactions, items)                    # 서비스용은 상호작용 기록 전체로 학습
    print(f"② 학습 — 상호작용 기록 {len(interactions):,}건 · 투자자 {len(묶음['사용자_목록'])}명")

    # ③ 일괄 추천 — 300명 모두의 추천을 표 하나로 만든다
    표 = 일괄_추천(묶음, users, items)
    print(f"③ 일괄 추천 — {len(표):,}줄")

    # ④ 규격 검사 — 어긴 것이 있으면 저장하지 않고 멈춘다 (앞의 5개만 화면에 보여 준다)
    문제 = 규격_검사(표, users, 묶음)
    if 문제:
        raise SystemExit("④ 규격 검사 실패 — " + " / ".join(문제[:5]))
    print("④ 규격 검사 — 문제 0개 · 통과")

    # ⑤ 저장 — ① 과 ④ 를 모두 통과했을 때만 여기에 온다
    출력_파일.parent.mkdir(exist_ok=True)                    # output 폴더가 없으면 만든다. 있어도 오류가 나지 않는다
    표.to_csv(출력_파일, index=False, encoding="utf-8-sig")  # utf-8-sig = 엑셀에서 한글이 안 깨진다. index=False = 줄 번호는 저장하지 않는다
    print(f"⑤ 저장 — {출력_파일} · {표['user_id'].nunique()}명 × 10개")


if __name__ == "__main__":                                  # 명령 한 줄(python recommend_batch.py)로 실행했을 때만 main() 이 돈다.
    main()                                                  # 노트북에서 import 할 때는 돌지 않는다
