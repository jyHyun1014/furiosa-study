# https://dacon.io/competitions/open/235576/overview/description
# 서울시 따릉이 대여량 예측 경진대회

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
import time
from datetime import datetime

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


# 스케일링
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
test_csv = scaler.transform(test_csv)


# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(5))
model.add(Dense(1))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=32, 
                 validation_data=(x_valid, y_valid),
                 callbacks=[es],
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

##################### submission.csv 만들기 // count 컬럼에 값 넣어준다 #########################
y_submit = model.predict(test_csv)

submission['count'] = y_submit
# print(submission)
# print(submission.shape)

filename = datetime.now().strftime("submit_%m%d_%H%M.csv")
submission.to_csv(path + "submit/" + filename)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("따릉이 Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()

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

'''
# 2차 시도 submit_0910_1649

MinMaxScaler 적용

random_stat = 42
test_size=0.2
epochs = 500000
batch_size = 32

Early Stopping
monitor='val_loss',
mode='min',
patience=20,
restore_best_weights=True,

# 결과
R2 : 0.6045787782108221
MSE : 2882.8021325108125
RMSE : 53.69173244095233
걸린시간 :  13.59 초
# 데이콘 점수
RMSE : 69.1656847734
'''

'''
# 3차 시도 submit_0911_1330

StandardScaler 적용

random_stat = 42
test_size=0.2
epochs = 500000
batch_size = 32

Early Stopping
monitor='val_loss',
mode='min',
patience=20,
restore_best_weights=True,

# 결과
R2 : 0.6059294546352809
MSE : 2872.9550816642645
RMSE : 53.59995411998283
걸린시간 :  5.83 초
# 데이콘 점수
RMSE : 69.6505069905
'''


'''
#######################################
# 4차 시도 submit_0911_1529

MaxAbsScaler 적용

random_stat = 42
test_size=0.2
epochs = 500000
batch_size = 32

Early Stopping
monitor='val_loss',
mode='min',
patience=20,
restore_best_weights=True,

# 결과
R2 : 0.6039252799013126
MSE : 2887.5664350227557
RMSE : 53.73608131435298
걸린시간 :  21.26 초
# 데이콘 점수
RMSE : 69.6190073622
#######################################
'''


'''
#######################################
# 5차 시도 submit_0911_1626

RobustScaler 적용

random_stat = 42
test_size=0.2
epochs = 500000
batch_size = 32

Early Stopping
monitor='val_loss',
mode='min',
patience=20,
restore_best_weights=True,

# 결과
R2 : 0.6053325020078316
MSE : 2877.307140209914
RMSE : 53.64053635274273
걸린시간 :  8.85 초
# 데이콘 점수
RMSE : 
#######################################
'''