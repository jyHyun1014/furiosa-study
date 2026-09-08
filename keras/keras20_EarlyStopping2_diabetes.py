from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np
import time


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (309, 10) (66, 10) (67, 10) (309,) (66,) (67,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=10, 
                 validation_data=(x_val, y_val),
                 callbacks=[es],
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

# R2 : 0.4392057213977827
# MSE : 3184.177920976068
# RMSE : 56.42852045708861

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