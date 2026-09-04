# https://dacon.io/competitions/open/235576/overview/description
# 서울시 따릉이 대여량 예측 경진대회

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd

# 1. 데이터
# path = "./_data/ddarung/"   # 상대경로 # 현재위치는 study 디렉토리임
path = "c:/study/_data/ddarung/"    # 절대경로
# path = "c:\study\_data\ddarung/"    # 슬래시, 역슬래시 상관없음
# path = "c://study//_data//ddarung/"
# path = "c:\\study\\_data\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission.shape) # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

print(train_csv.info())
# Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

# exit()
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

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_valid, y_valid)
y_pred = model.predict(x_valid)

r2 = r2_score(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = root_mean_squared_error(y_valid, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)
# R2 : 0.6018080522565566
# MSE : 2903.0019959713977
# RMSE : 53.87951369464462

##################### submission.csv 만들기 // count 컬럼에 값 넣어준다 #########################
y_submit = model.predict(test_csv)

submission['count'] = y_submit
# print(submission)
# print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0904_1149.csv")


'''
# 1차 시도 submit_0904_1149
random_stat = 42
test_size=0.2
epochs = 1000
batch_size = 32
# 결과
R2 : 0.578619459048384
MSE : 3072.0574797615077
RMSE : 55.42614437033761
# 데이콘 점수
RMSE : 69.4223819395
'''