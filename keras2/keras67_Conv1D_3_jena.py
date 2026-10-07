# 58-1 카피
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # 메모리 모으기

import pandas as pd
import numpy as np
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

data = pd.read_csv("./_data/kaggle_jena/jena_climate_2009_2016.csv", index_col=0)

print(data.head())
#                      p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# Date Time                                                                                                                                                                                    
# 01.01.2009 00:10:00    996.52     -8.02    265.40        -8.90    93.3          3.33          3.11          0.22       1.94             3.12       1307.75      1.03           1.75     152.3
# 01.01.2009 00:20:00    996.57     -8.41    265.01        -9.28    93.4          3.23          3.02          0.21       1.89             3.03       1309.80      0.72           1.50     136.1
# 01.01.2009 00:30:00    996.53     -8.51    264.91        -9.31    93.9          3.21          3.01          0.20       1.88             3.02       1310.24      0.19           0.63     171.6
# 01.01.2009 00:40:00    996.51     -8.31    265.12        -9.07    94.2          3.26          3.07          0.19       1.92             3.08       1309.19      0.34           0.50     198.0
# 01.01.2009 00:50:00    996.51     -8.27    265.15        -9.04    94.1          3.27          3.08          0.19       1.92             3.09       1309.00      0.32           0.63     214.3
print(data.tail())
#                      p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# Date Time                                                                                                                                                                                    
# 31.12.2016 23:20:00   1000.07     -4.05    269.10        -8.13   73.10          4.52          3.30          1.22       2.06             3.30       1292.98      0.67           1.52     240.0
# 31.12.2016 23:30:00    999.93     -3.35    269.81        -8.06   69.71          4.77          3.32          1.44       2.07             3.32       1289.44      1.14           1.92     234.3
# 31.12.2016 23:40:00    999.82     -3.16    270.01        -8.21   67.91          4.84          3.28          1.55       2.05             3.28       1288.39      1.08           2.00     215.2
# 31.12.2016 23:50:00    999.81     -4.23    268.94        -8.53   71.80          4.46          3.20          1.26       1.99             3.20       1293.56      1.49           2.16     225.8
# 01.01.2017 00:00:00    999.82     -4.82    268.36        -8.42   75.70          4.27          3.23          1.04       2.01             3.23       1296.38      1.23           1.96     184.9


# 0도와 360도 방향의 순환 특성 표현하기
# x_test 데이터 144개: 2016.12.30 00:10:00 ~ 2016.12.31 00:00:00 
# 예측할 데이터 144개: 2016.12.31 00:10:00 ~ 2017.01.01 00:00:00 기간의 T (degC)
# x.shape (N, 144, 13)
# y.shape (N, 144)

wd = np.deg2rad(data["wd (deg)"].to_numpy(dtype=np.float32))
data["wd_sin"] = np.sin(wd)
data["wd_cos"] = np.cos(wd)
data = data.drop(columns=["wd (deg)"])

## 훈련할 데이터 자르기 ###
x_train = data[:-288].drop(["T (degC)"], axis=1).to_numpy(dtype=np.float32)
y_train = data[144:-144]["T (degC)"].to_numpy(dtype=np.float32)

x_test = data[-288:-144].drop(["T (degC)"], axis=1)
y_test = data[-144:]["T (degC)"]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (420263, 13) (420263,) (144, 13) (144,)

# 스케일링
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 시계열 데이터로 만들기
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10-3+1 = range(0, 8)
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

timestep = 144

x_train = split_x(x_train, timestep)
y_train = split_x(y_train, timestep)

x_test = split_x(x_test, timestep)
y_test = split_x(y_test, timestep)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (420120, 144, 13) (420120, 144) (1, 144, 13) (1, 144)
n_train, seq_len, n_features = x_train.shape

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape) # (336096, 144, 14) (336096, 144) (84024, 144, 14) (84024, 144) (1, 144, 14) (1, 144)

####################################

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, input_shape=(144, 14)))
model.add(Conv1D(128, 2))
model.add(Conv1D(32, 2))
model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(64, activation='relu'))
model.add(Dense(144))

model.summary()

# 3. 컴파일
es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)
path = "./_save/keras67/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k67_03_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, 
          epochs=100,
          batch_size=128,
          verbose=1,
          validation_data=(x_val, y_val), # validation_split=0.2는 배열의 마지막 20%를 먼저 검증 세트로 분리하고, 나머지 학습 데이터만 섞기 때문에 train_test_split으로 섞음과 동시에 검증데이터 분리
          shuffle=True,
          callbacks=[es, mcp],
          )
end_time = time.time()

# 4. 평가, 예측
print("=====================================")
results= model.evaluate(x_test, y_test)
print("loss :", results)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# y_test.shape (1, 144)라서 r2_score에 넣기 위해 1차원으로 펼침
y_true = y_test.reshape(-1)
y_pred = model.predict(x_test).reshape(-1)

r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse) # 목표: 1.48

# Epoch 12: early stopping
# =====================================
# 1/1 [==============================] - 0s 390ms/step - loss: 5.0683
# loss : 5.068253040313721
# 걸린시간 :  544.63 초
# 1/1 [==============================] - 0s 341ms/step
# R2 : 0.5348142426082619
# MSE : 5.068253257734142
# RMSE : 2.251278138687919