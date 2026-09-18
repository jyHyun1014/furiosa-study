# 33-3 카피

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import boston_housing
import time
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (404, 13) (102, 13) (404,) (102,)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# CNN 모델에 넣기 위해 reshape
x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)
print(x_train.shape, y_train.shape) # (404, 13, 1, 1) (404,)
print(x_test.shape, y_test.shape) # (102, 13, 1, 1) (102,)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(16, (2,1), activation='relu', input_shape=(13, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)

model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
print("=====================================")
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

"""
##########################################
CNN 모델
RobustScaler 적용
Dropout 적용

R2 : 0.6832346615914473
MSE : 26.368765846804767
RMSE : 5.135052662515231
걸린시간 :  7.28 초
###########################################
"""