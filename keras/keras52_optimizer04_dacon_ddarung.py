# 35-4 카피

# https://dacon.io/competitions/open/235576/overview/description
# 서울시 따릉이 대여량 예측 경진대회

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pandas as pd
import time
import datetime

# 1. 데이터
path = "c:/study/_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission.shape) # (715, 1)

############################# 결측치 처리 1. 삭제 #############################
train_csv = train_csv.dropna()

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1)   # 열(컬럼) 삭제
y = train_csv['count']

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)   # (1062, 9) (266, 9) (1062,) (266,)

################## submit 물밑작업 ##################
print(test_csv.info())
# Index: 715 entries, 0 to 2177
# Data columns (total 9 columns):
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    715 non-null    int64  
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64

################## 결측치 처리 2. 평균값 넣기 ##################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
test_csv = scaler.transform(test_csv)


# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dropout(0.4))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=100, 
                 batch_size=32, 
                 validation_data=(x_valid, y_valid),
                 )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
loss = model.evaluate(x_valid, y_valid)
y_pred = model.predict(x_valid)

r2 = r2_score(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = root_mean_squared_error(y_valid, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초") # 2번째 자리에서 반올림

# CPU
# R2 : 0.5989610093635696
# MSE : 2923.7582449307138
# RMSE : 54.071787883615556
# 걸린시간 :  9.4 초

# GPU
# R2 : 0.5991660441829619
# MSE : 2922.2634470240437
# RMSE : 54.05796377060501
# 걸린시간 :  7.86 초

# learning_rate=0.01
# R2 : 0.6046641758859121
# MSE : 2882.179544277568
# RMSE : 53.68593432434205
# 걸린시간 :  7.65 초