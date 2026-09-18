# 33-2 복사

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import datetime

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (309, 10) (133, 10) (309,) (133,)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# CNN 모델에 넣기 위해 reshape
x_train = x_train.reshape(-1, 10, 1, 1)
x_test = x_test.reshape(-1, 10, 1, 1)
print(x_train.shape, y_train.shape) # (309, 10, 1, 1) (309,)
print(x_test.shape, y_test.shape) # (133, 10, 1, 1) (133,)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (2,1), activation='relu', input_shape=(10, 1, 1)))
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
    patience=50,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)
model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=10, 
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

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("당뇨병 Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()

"""
##########################################
MaxAbsScaler 적용

R2 : 0.37587622615402116
MSE : 3543.7614406303055
RMSE : 59.52950059113805
걸린시간 :  5.46 초
###########################################
"""

"""
##########################################
RobustScaler 적용

R2 : 0.42711149528445147
MSE : 3252.848678205168
RMSE : 57.03375034315356
걸린시간 :  7.38 초
###########################################
"""

"""
##########################################
RobustScaler 적용 (ModelCheckpoint)

R2 : 0.43912842320208056
MSE : 3184.616818478196
RMSE : 56.43240929180851
걸린시간 :  8.01 초
###########################################
"""

"""
##########################################
RobustScaler 적용 (ModelCheckpoint)
Dropout 적용

R2 : 0.4230040234645377
MSE : 3276.1708153578516
RMSE : 57.237844258478596
걸린시간 :  10.05 초
###########################################
"""

"""
##########################################
CNN 모델
RobustScaler 적용
Dropout 적용

R2 : 0.4838373288691703
MSE : 2786.4049539613266
RMSE : 52.786408799626884
걸린시간 :  12.38 초
###########################################
"""