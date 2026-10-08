# 53-2 복사

# 회귀 문제 Conv1D 모델링

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D, Flatten, GlobalAveragePooling1D
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
model.add(Reshape(target_shape=(10, 1), input_shape=(10,)))
model.add(Conv1D(filters=10, kernel_size=2, padding="same", input_shape=(10, 1)))
model.add(Conv1D(10, 2, padding="same"))

model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(16))
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

# R2 : 0.3850154252111768
# MSE : 3491.869263831116
# RMSE : 59.092040613191855
# 걸린시간 :  5.4 초