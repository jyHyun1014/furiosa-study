# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # 메모리 모으기

import pandas as pd
import numpy as np
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
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

# x_test 데이터 144개: 2016.12.30 00:10:00 ~ 2016.12.31 00:00:00 
# 예측할 데이터 144개: 2016.12.31 00:10:00 ~ 2017.01.01 00:00:00 기간의 T (degC)
# x.shape (N, 144, 13)
# y.shape (N, 144)

y_cor = data[-144:]["T (degC)"] # 예측치 정답 데이터
print(y_cor)
# Date Time
# 31.12.2016 00:10:00    176.3
# 31.12.2016 00:20:00    179.8
# 31.12.2016 00:30:00    182.7
# 31.12.2016 00:40:00    200.6
# 31.12.2016 00:50:00    203.8
#                        ...  
# 31.12.2016 23:20:00    240.0
# 31.12.2016 23:30:00    234.3
# 31.12.2016 23:40:00    215.2
# 31.12.2016 23:50:00    225.8
# 01.01.2017 00:00:00    184.9

### 훈련할 데이터 자르기 ###
x_data = data[:-288].drop(["T (degC)"], axis=1).to_numpy(dtype=np.float32)
y_data = data[144:-144]["T (degC)"].to_numpy(dtype=np.float32)
print(x_data.shape) # (420263, 13)
print(y_data.shape) # (420263,)

size_x = 144
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10-3+1 = range(0, 8)
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

start_time = time.time()
x = split_x(x_data, size_x)
y = split_x(y_data, size_y)
end_time = time.time()

# submit용 x데이터
x_predict = data[-288:-144].drop(["T (degC)"], axis=1)
print(type(x_predict)) # <class 'pandas.DataFrame'>
x_predict = x_predict.to_numpy()
print(x_predict.shape) # (144, 13)
x_predict = x_predict.reshape(1,144,13)

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (336096, 144, 13) (336096, 144) (84024, 144, 13) (84024, 144)


# 스케일링을 위한 reshape
n_train, seq_len, n_features = x_train.shape
n_test = x_test.shape[0]

x_train = x_train.reshape(-1, n_features) # (336096 * 144, 13)
x_test = x_test.reshape(-1, n_features) # (84024 * 144, 13)
x_predict = x_predict.reshape(-1, n_features)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

# 모델에 넣기 위한 reshape
x_train = x_train.reshape(n_train, seq_len, n_features)
x_test = x_test.reshape(n_test, seq_len, n_features)
x_predict = x_predict.reshape(1, seq_len, n_features)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (336096, 144, 13) (336096, 144) (84024, 144, 13) (84024, 144)

####################################

# 2. 모델 구성
model = Sequential()
model.add(LSTM(units=32, input_shape=(144, 13), return_sequences=True))
# 모든 timestep의 hidden state를 다음 RNN layer에 전달하기 위해 3차원으로 출력
# return_sequences=False(기본값)이면 마지막 timestep의 hidden state만 출력하여 2차원이 됨

model.add(LSTM(32, return_sequences=True))
# model.add(LSTM(64, return_sequences=True))

model.add(Flatten())
# LSTM의 3차원 출력 데이터를 2차원으로 펼침. Dense layer에 전달하기 위해
# (batch_size, 3, 5) → (batch_size, 15)

model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(144))

model.summary()

# 3. 컴파일
es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=30,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)
path = "./_save/keras58/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k58_02_", date, filename]) # 모델 저장 경로

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
          epochs=1,
          batch_size=128,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[es, mcp],
          )
end_time = time.time()

# 4. 평가, 예측
print("=====================================")
results= model.evaluate(x_test, y_test)
print("loss :", results)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)


y_submit = model.predict(x_predict)
print(y_submit)
print(y_submit.shape)

# loss : 5.412315368652344
# 걸린시간 :  52.69 초
# 2626/2626 [==============================] - 20s 8ms/step
# R2 : 0.9233404994010925
# MSE : 5.412240028381348
# RMSE : 2.2677090167999268