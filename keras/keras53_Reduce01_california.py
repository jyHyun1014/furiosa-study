# learning_rate를 바꿔가며 학습시켜보기

# 52-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (14448, 8) (3096, 8) (3096, 8) (14448,) (3096,) (3096,)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) # -0.0010638297872338498 1.333173652694611

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_shape=(8,)))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dropout(0.4))
model.add(Dense(6))
model.add(Dropout(0.4))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau( # val_loss가 10 epoch 동안 개선되지 않으면 Learning Rate를 절반으로 줄임
    monitor="val_loss",
    mode="auto",
    patience=10,
    verbose=1,
    factor=0.5,
)

model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=1000, batch_size=32, validation_data=(x_val, y_val),
                 callbacks=[es, reduce_lr],
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

# CPU
# R2 : 0.5693080280032757
# MSE : 0.5695084829073579
# RMSE : 0.7546578581763778
# 걸린시간 :  46.95 초

# GPU
# R2 : 0.6025316154076732
# MSE : 0.5255765870521867
# RMSE : 0.7249666109912833
# 걸린시간 :  98.27 초

# learning_rate = 0.0001
# R2 : 0.3052392354983552
# MSE : 0.9186893991557784
# RMSE : 0.9584828632561869
# 걸린시간 :  87.43 초

# learning_rate = 0.01
# es patience=20
# reduce_lr patience=10
# R2 : 0.4010256859138339
# MSE : 0.7920299775596519
# RMSE : 0.8899606606809382
# 걸린시간 :  35.77 초