# 아모레 모델
# 하이닉스 모델
# 삼성전자 모델(시가 컬럼 제외)

# 각 모델에 컬럼 5개 이상
# 각 모델에 LSTM, Conv1D 필수로 넣기

# 앙상블 활용해서 3가지 모델을 하나로 concatenate

# 삼성전자의 내일 시가 맞추기

# 세 종목의 최근 10거래일 데이터를 입력하고 다음 거래일 삼성전자 시가를 정답으로 구성

import numpy as np
import pandas as pd
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# 1. 데이터
path = "./_data/samsung/"
amore_df = pd.read_csv(path + "아모레261012.csv", encoding="cp949")
hynix_df = pd.read_csv(path + "하이닉스261012.csv", encoding="cp949")
samsung_df = pd.read_csv(path + "삼성전자261012.csv", encoding="cp949")

df_list = [amore_df, hynix_df, samsung_df]

# 날짜 정렬 및 숫자 변환
for df in df_list:
    df.rename(columns={"Unnamed: 6": "전일종가대비"}, inplace=True) # "Unnamed: 6" 열의 이름을 "전일종가대비"로 변경
    # print(df["대비"].unique())
    df["대비"] = df["대비"].str.strip().map({"▲": 1, "▼": -1, "↑": 1, "↓": -1, "": 0,}) # 상승·하락 기호를 숫자로 변환
    df["일자"] = pd.to_datetime(df["일자"]) # "일자" 열을 datetime 타입으로 변환
    df.set_index("일자", inplace=True) # "일자" 열을 행 인덱스로 설정
    df.sort_index(inplace=True) # 날짜 인덱스를 오름차순으로 정렬: 과거 → 최근

# 데이터의 각 컬럼을 숫자로 변환
for df in df_list:
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )


# 회사별로 열 이름에 접두사 추가 후 날짜 인덱스를 기준으로 병합
merged_df = pd.concat(
    [
        amore_df.add_prefix("아모레_"),
        hynix_df.add_prefix("하이닉스_"),
        samsung_df.add_prefix("삼성전자_"),
    ],
    axis=1,        # 열 방향으로 합치기
    join="inner",  # 세 회사에 모두 존재하는 날짜만 사용
).sort_index()    # 과거 → 최근 순으로 정렬

# print(merged_df.head())
print(merged_df.shape) # (1120, 48)

# 결측치가 있는 열과 개수 확인
# missing = merged_df.isna().sum()
# print(missing[missing > 0])


# 최근 10거래일 → 다음 거래일 삼성전자 시가
window_size = 10
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

# 병합된 데이터에서 회사별 열 선택 후 DataFrame을 NumPy 배열로 변환
amore_data = merged_df.filter(regex=r"^아모레_").to_numpy(dtype=np.float32) # 열 이름이 "아모레_"로 시작하는 열만 선택
hynix_data = merged_df.filter(regex=r"^하이닉스_").to_numpy(dtype=np.float32)
samsung_data = merged_df.filter(regex=r"^삼성전자_").drop(columns="삼성전자_시가").to_numpy(dtype=np.float32)

# 회사별로 11거래일씩 묶기: 입력 10일 + 정답 1일
amore_windows = split_x(amore_data, window_size + 1)
hynix_windows = split_x(hynix_data, window_size + 1)
samsung_windows = split_x(samsung_data, window_size + 1)

# 회사별 입력 x: 각 묶음의 앞 10거래일, 모든 특성
amore_x = amore_windows[:, :-1, :]
hynix_x = hynix_windows[:, :-1, :]
samsung_x = samsung_windows[:, :-1, :]

# 공통 정답 y: 다음 거래일 삼성전자 시가
y = merged_df["삼성전자_시가"].to_numpy(dtype=np.float32)[window_size:]

# 다음 거래일 예측용 입력: 회사별 최근 10거래일
amore_predict = amore_data[-window_size:]
hynix_predict = hynix_data[-window_size:]
samsung_predict = samsung_data[-window_size:]
amore_predict = np.expand_dims(amore_predict, axis=0) # (10, features) → (1, 10, features)
hynix_predict = np.expand_dims(hynix_predict, axis=0) # (10, features) → (1, 10, features)
samsung_predict = np.expand_dims(samsung_predict, axis=0) # (10, features) → (1, 10, features)

print(amore_x.shape, hynix_x.shape, samsung_x.shape, y.shape) # (1110, 10, 16) (1110, 10, 16) (1110, 10, 15) (1110,)
print(amore_predict.shape, hynix_predict.shape, samsung_predict.shape, y.shape) # (1, 10, 16) (1, 10, 16) (1, 10, 15) (1110,)

####################################

# 윈도우 내부 날짜 순서는 유지하고 샘플을 무작위 분할
# 10거래일 묶음 단위로 무작위 분할
amore_x_train, amore_x_val, hynix_x_train, hynix_x_val, samsung_x_train, samsung_x_val, y_train, y_val = train_test_split(amore_x, hynix_x, samsung_x, y,
                                                                                                                          test_size=0.2, shuffle=True, random_state=42)

# 2. Functional API: 종목별 Conv1D + LSTM 분기
input1 = Input(shape=amore_x.shape[1:], name="amore_input")
conv1 = Conv1D(32, kernel_size=2, padding="same", activation="relu", name="amore_conv")(input1)
lstm1 = LSTM(32, name="amore_lstm")(conv1)
dense1 = Dense(16, activation="relu", name="amore_dense")(lstm1)

input2 = Input(shape=hynix_x.shape[1:], name="hynix_input")
conv2 = Conv1D(32, kernel_size=2, padding="same", activation="relu", name="hynix_conv")(input2)
lstm2 = LSTM(32, name="hynix_lstm")(conv2)
dense2 = Dense(16, activation="relu", name="hynix_dense")(lstm2)

input3 = Input(shape=samsung_x.shape[1:], name="samsung_input")
conv3 = Conv1D(32, kernel_size=2, padding="same", activation="relu", name="samsung_conv")(input3)
lstm3 = LSTM(32, name="samsung_lstm")(conv3)
dense3 = Dense(16, activation="relu", name="samsung_dense")(lstm3)

# 세 종목의 특징을 합쳐 다음 거래일 시가를 출력
merged = concatenate([dense1, dense2, dense3], name="merge_stocks")
merged = Dense(32, activation="relu", name="merged_dense1")(merged)
merged = Dense(16, activation="relu", name="merged_dense2")(merged)
output = Dense(1, name="next_samsung_open")(merged)

model = Model(inputs=[input1, input2, input3], outputs=output, name="stock_ensemble")

model.summary()


# 3. 컴파일
es = EarlyStopping( # 10 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

path = "./_save/keras70/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k70_01_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

start_time = time.time() # 현재시간을 반환. 시작시간
model.fit([amore_x_train, hynix_x_train, samsung_x_train],
          y_train,
          epochs=10000000, 
          batch_size=64,
          validation_data=([amore_x_val, hynix_x_val, samsung_x_val], y_val),
          callbacks=[es, mcp]
          )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
print("=====================================")

val_loss, val_mae = model.evaluate([amore_x_val, hynix_x_val, samsung_x_val], y_val) # 검증 데이터 평가

y_pred = model.predict([amore_x_val, hynix_x_val, samsung_x_val]).ravel() # 검증 데이터 예측 # ravel: (샘플 수, 1) → (샘플 수,)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"검증 MSE: {val_loss:,.2f}")
print(f"검증 MAE: {val_mae:,.2f}")
print(f"검증 RMSE: {rmse:,.2f}")
print(f"검증 R²: {r2:.4f}")
print(f"학습 시간: {end_time - start_time:.2f}초")


# 회사별 최근 10거래일을 입력하여 다음 거래일 시가 예측
next_open = model.predict(
    [amore_predict, hynix_predict, samsung_predict],
    verbose=0,
).ravel()[0]

print("마지막 입력 날짜 :", merged_df.index[-1].strftime("%Y-%m-%d"))
print("다음 거래일 삼성전자 예상 시가 :", f"{next_open:,.0f}원")

# 검증 MSE: 248,839,936.00
# 검증 MAE: 10,594.72
# 검증 RMSE: 15,774.66
# 검증 R²: 0.9395
# 학습 시간: 89.63초
# 마지막 입력 날짜 : 2026-10-12
# 다음 거래일 삼성전자 예상 시가 : 235,296원
