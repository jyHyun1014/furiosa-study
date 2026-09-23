# 52-3 카피

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import boston_housing
import time
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (404, 13) (102, 13) (404,) (102,)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (404, 13) (51, 13) (51, 13) (404,) (51,) (51,)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

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
                 batch_size=32, 
                 validation_data=(x_val, y_val),
                 callbacks=[es, reduce_lr],
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

# CPU
# R2 : 0.6312003040750593
# MSE : 20.975491922178612
# RMSE : 4.579900863793736
# 걸린시간 :  7.03 초

# GPU
# R2 : 0.6391220392853716
# MSE : 20.524943034124863
# RMSE : 4.530446229029196
# 걸린시간 :  4.45 초

# learning_rate=0.01
# R2 : 0.6307813850390251
# MSE : 20.99931795282201
# RMSE : 4.5825012769034785
# 걸린시간 :  4.81 초

# learning_rate = 0.01
# es patience=20
# reduce_lr patience=10
# R2 : 0.632571357420451
# MSE : 20.897513228896305
# RMSE : 4.571379794864599
# 걸린시간 :  2.28 초