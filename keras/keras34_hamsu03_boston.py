# 33-3 카피

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import boston_housing
import time
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (404, 13) (102, 13) (404,) (102,)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (404, 13) (51, 13) (51, 13) (404,) (51,) (51,)

# 스케일링
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(16, input_dim=13))
# model.add(Dropout(0.4))
# model.add(Dense(10))
# model.add(Dense(6))
# model.add(Dense(1))

input1 = Input(shape=(13,))
dense1 = Dense(16, name='jy1')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(10, name='jy2')(drop1)
dense3 = Dense(6, name='jy3')(dense2)
output1 = Dense(1, name='jy4')(dense3)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)

path = "./_save/keras33/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k33_03_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)

model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=32, 
                 validation_data=(x_val, y_val),
                 callbacks=[es, mcp],
                 )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초") # 2번째 자리에서 반올림

# R2 : 0.6501915605442139
# MSE : 19.89536373589657
# RMSE : 4.460421923528823
# 걸린시간 :  4.97 초

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("보스턴 집값 Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()

"""
##########################################
MaxAbsScaler 적용

R2 : 0.57785638603037
MSE : 24.009428594055542
RMSE : 4.899941692924064
걸린시간 :  14.5 초
###########################################
"""

"""
##########################################
RobustScaler 적용

R2 : 0.5825583213124677
MSE : 23.742005907383056
RMSE : 4.8725769267794075
걸린시간 :  4.78 초
###########################################
"""

"""
##########################################
RobustScaler 적용 (ModelCheckpoint)

R2 : 0.6039892753143348
MSE : 22.523119862958996
RMSE : 4.745852912065333
걸린시간 :  6.65 초
###########################################
"""

"""
##########################################
RobustScaler 적용 (ModelCheckpoint)
Dropout 적용

R2 : 0.6078116715264746
MSE : 22.30572098287038
RMSE : 4.722893285145281
걸린시간 :  8.52 초
###########################################
"""