# 29-3 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (14448, 8) (3096, 8) (3096, 8) (14448,) (3096,) (3096,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) # -0.0010638297872338498 1.333173652694611

# 2. 모델구성
# model = Sequential()
# model.add(Dense(16, input_dim=8))
# model.add(Dense(10))
# model.add(Dense(6))
# model.add(Dense(1))

# model.summary()

path = "./_save/keras29/"
# model.save(path + "keras29_1_save_model.keras") # 모델 구조 + 초기 가중치 저장
model = load_model(path + "keras29_3_save_model.keras")

model.summary()

# exit()

# 3. 컴파일, 훈련
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=30,
#     restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
# )

# model.compile(loss='mse', optimizer='adam')

# start_time = time.time() # 현재시간을 반환. 시작시간
# hist = model.fit(x_train, y_train, epochs=100000, batch_size=32, validation_data=(x_val, y_val), callbacks=[es])
# end_time = time.time() # 현재시간을 반환. 끝시간

# model.save(path + "keras29_3_save_model.keras") # 모델 구조 + 학습된 가중치 저장

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

# R2 : 0.6149186264263757
# MSE : 0.509197113294381
# RMSE : 0.7135804883083484
# 걸린시간 :  53.87 초


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


"""
##########################################
MaxAbsScaler 적용

R2 : 0.5351104873090005
MSE : 0.6147282473475167
RMSE : 0.784046074760608
걸린시간 :  13.84 초
###########################################
"""

"""
##########################################
RobustScaler 적용

R2 : 0.42375973554254787
MSE : 0.7619685068190551
RMSE : 0.8729080746671182
걸린시간 :  14.27 초
###########################################
"""