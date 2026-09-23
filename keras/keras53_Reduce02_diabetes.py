# 52-2 복사

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import datetime

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (309, 10) (66, 10) (67, 10) (309,) (66,) (67,)

# 스케일링

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
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
                 epochs=1000, 
                 batch_size=10, 
                 validation_data=(x_val, y_val),
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
# R2 : 0.4267852951916361
# MSE : 3254.7008353561864
# RMSE : 57.04998541065709
# 걸린시간 :  8.53 초

# GPU
# R2 : 0.4245385140996468
# MSE : 3267.4580103477197
# RMSE : 57.16168306083822
# 걸린시간 :  7.63 초

# learning_rate=0.01
# R2 : 0.418931550805131
# MSE : 3299.294228025654
# RMSE : 57.439483180349505
# 걸린시간 :  7.64 초

# learning_rate = 0.01
# es patience=20
# reduce_lr patience=10
# R2 : 0.42738078979722594
# MSE : 3251.3196297206623
# RMSE : 57.02034399861739
# 걸린시간 :  3.27 초