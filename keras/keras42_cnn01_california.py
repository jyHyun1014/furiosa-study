# 33-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (16512, 8) (4128, 8) (16512,) (4128,)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # -7.655796326968436 1461.5716128615384
print(np.min(x_test), np.max(x_test)) # -4.092435274420402 352.9563421908926

# CNN 모델에 넣기 위해 reshape
x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)
print(x_train.shape, y_train.shape) # (16512, 8, 1, 1) (16512,)
print(x_test.shape, y_test.shape) # (4128, 8, 1, 1) (4128,)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(16, (2,1), activation='relu', input_shape=(8, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=6, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# exit()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

model.compile(loss='mse', optimizer='adam')

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=100000, batch_size=32, validation_split=0.2, 
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

"""
##########################################
RobustScaler 적용
Dropout 적용

R2 : 0.32498846058487596
MSE : 0.8925747930128388
RMSE : 0.9447617652153578
###########################################
"""


"""
##########################################
CNN 모델
RobustScaler 적용
Dropout 적용

R2 : 0.7447528478814001
MSE : 0.33447815980663403
RMSE : 0.578340868179514
걸린시간 :  213.86 초
###########################################
"""