from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape,y_test.shape) # (14448, 8) (3096, 8) (3096, 8) (14448,) (3096,) (3096,)

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
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
                 validation_split=0.2,
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

# R2 : 0.5444869522611198
# MSE : 0.6023296500271189
# RMSE : 0.7760989949916949
# 걸린시간 :  4.74 초

print("===================== history ===========================")
print(hist)
# <keras.src.callbacks.history.History object at 0x0000017DFECE9150>
print("===================== history ===========================")
print(hist.history)
# {'loss': [163.20236206054688, 33.15802001953125, 9.632168769836426, 13.144063949584961, 4.1305131912231445, 62.19902038574219, 0.9429430961608887, 0.906912088394165, 1.7514152526855469, 1.2130557298660278], 
# 'val_loss': [15.313130378723145, 4.135632514953613, 0.7012370228767395, 0.7204505205154419, 1.3476210832595825, 0.8006155490875244, 0.8084685802459717, 0.9987784028053284, 0.7813654541969299, 1.2771917581558228]}
print("===================== loss ===========================")
print(hist.history['loss'])
print("===================== val_loss ===========================")
print(hist.history['val_loss'])
print("================================================")


import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("캘리포니아 Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()