# 저장된 모델과 원본 CSV를 불러와 최근 10거래일로 다음 거래일 삼성전자 시가 예측

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# 1. 경로 설정
data_path = "./_data/samsung/"

# 실제 사용할 모델 파일명으로 변경
model_path = "./_save/keras70/k70_01_1013_1056_0459-248839936.0000.keras"

window_size = 10


# 2. 모델 불러오기
# 예측만 하므로 컴파일 설정은 불러오지 않음
model = load_model(model_path, compile=False)


# 3. 원본 CSV 불러오기
amore_df = pd.read_csv(
    data_path + "아모레261012.csv", encoding="cp949"
)
hynix_df = pd.read_csv(
    data_path + "하이닉스261012.csv", encoding="cp949"
)
samsung_df = pd.read_csv(
    data_path + "삼성전자261012.csv", encoding="cp949"
)

df_list = [amore_df, hynix_df, samsung_df]


# 4. 학습 때와 동일한 전처리
for df in df_list:
    df.rename(
        columns={"Unnamed: 6": "전일종가대비"},
        inplace=True,
    )

    df["대비"] = df["대비"].str.strip().map({
        "▲": 1,
        "▼": -1,
        "↑": 1,
        "↓": -1,
        "": 0,
    })

    df["일자"] = pd.to_datetime(df["일자"])
    df.set_index("일자", inplace=True)
    df.sort_index(inplace=True)

    # 날짜가 중복되면 날짜 기준 병합이 올바르지 않을 수 있음
    if not df.index.is_unique:
        raise ValueError("CSV에 중복된 날짜가 있습니다.")

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )


# 5. 세 회사의 공통 날짜를 기준으로 병합
merged_df = pd.concat(
    [
        amore_df.add_prefix("아모레_"),
        hynix_df.add_prefix("하이닉스_"),
        samsung_df.add_prefix("삼성전자_"),
    ],
    axis=1,
    join="inner",
).sort_index()

if len(merged_df) < window_size:
    raise ValueError("공통 날짜 기준으로 최소 10거래일이 필요합니다.")


# 6. 가장 최근 10거래일 선택
recent_df = merged_df.iloc[-window_size:]

# 학습 때와 동일한 회사별 열 선택 및 순서 유지
amore_data = recent_df.filter(
    regex=r"^아모레_"
).to_numpy(dtype=np.float32)

hynix_data = recent_df.filter(
    regex=r"^하이닉스_"
).to_numpy(dtype=np.float32)

# 학습 때 삼성전자 시가를 제외했으므로 예측에서도 제외
samsung_data = (
    recent_df.filter(regex=r"^삼성전자_")
    .drop(columns="삼성전자_시가")
    .to_numpy(dtype=np.float32)
)

for name, data in [
    ("아모레", amore_data),
    ("하이닉스", hynix_data),
    ("삼성전자", samsung_data),
]:
    if not np.isfinite(data).all():
        raise ValueError(f"{name}의 최근 입력에 NaN 또는 무한대가 있습니다.")


# 7. 배치 차원 추가
amore_predict = np.expand_dims(amore_data, axis=0)
hynix_predict = np.expand_dims(hynix_data, axis=0)
samsung_predict = np.expand_dims(samsung_data, axis=0)

# 예상 shape: (1, 10, 16), (1, 10, 16), (1, 10, 15)
predict_inputs = [amore_predict, hynix_predict, samsung_predict]

# 불러온 모델과 입력 크기가 일치하는지 확인
for model_input, data in zip(model.inputs, predict_inputs):
    expected_shape = tuple(model_input.shape[1:])

    if data.shape[1:] != expected_shape:
        raise ValueError(
            f"{model_input.name}: "
            f"모델 입력 크기 {expected_shape}, "
            f"현재 데이터 크기 {data.shape[1:]}"
        )


# 8. 다음 거래일 삼성전자 시가 예측
next_open = model.predict(
    predict_inputs,
    verbose=0,
).ravel()[0]


# 9. 결과 출력
print("불러온 모델 :", model_path)
print("입력 시작 날짜 :", recent_df.index[0].strftime("%Y-%m-%d"))
print("마지막 입력 날짜 :", recent_df.index[-1].strftime("%Y-%m-%d"))
print("다음 거래일 삼성전자 예상 시가 :", f"{next_open:,.0f}원")