# 69-3 카피

# 하나의 모델을 2개의 모델로 분할하여 2개의 값을 예측하는 회귀 모델

import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

# 1. 데이터

# 첫 번째 입력 데이터: 삼성전자 종가, SK하이닉스 종가
x1_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)
print(x1_datasets)

# 두 번째 입력 데이터: 원유가, 환율, 금시세
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3)

# 세 번째 입력 데이터
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).transpose() # (100, 4)

# 첫번째 예측 대상: 화성의 화씨 온도
y1 = np.array(range(3001, 3101))

# 두번째 예측 대상: 비트코인 가격
y2 = np.array(range(13001, 13101))


# x_datasets = np.concatenate(
#     [x1_datasets, x2_datasets, x3_datasets],
#     axis=1
# )
# print(x_datasets.shape)  # (100, 9)

# x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x_datasets, y1, y2, test_size=0.2, random_state=42)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2, test_size=0.2, random_state=42)

# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델 분기1
last_dense1 = Dense(10, name="ld1")(output1)
last_dense2 = Dense(10, name="ld2")(last_dense1)
last_output1 = Dense(1, name="last1")(last_dense2)

# 2-3. 모델 분기2
last_dense21 = Dense(10, name="ld21")(output1)
last_dense22 = Dense(10, name="ld22")(last_dense21)
last_output2 = Dense(1, name="last21")(last_dense22)


model = Model(inputs=[input1], 
              outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일
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

model.compile(loss="mse", optimizer="adam")
start_time = time.time() # 현재시간을 반환. 시작시간
model.fit([x1_train], 
          [y1_train, y2_train], 
          epochs=10000, 
          batch_size=1,
          validation_split=0.2,
          callbacks=[es, reduce_lr]
          )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
print("=====================================")
result = model.evaluate([x1_test], [y1_test, y2_test])
print("loss :", result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T
y1_pred, y2_pred = model.predict([x1_pred])

y1_pred, y2_pred = model.predict([x1_test])

# 첫 번째 출력 평가
r2_1 = r2_score(y1_test, y1_pred)
mse_1 = mean_squared_error(y1_test, y1_pred)
rmse_1 = root_mean_squared_error(y1_test, y1_pred)

# 두 번째 출력 평가
r2_2 = r2_score(y2_test, y2_pred)
mse_2 = mean_squared_error(y2_test, y2_pred)
rmse_2 = root_mean_squared_error(y2_test, y2_pred)

print("화성 온도")
print("R2 :", r2_1)
print("MSE :", mse_1)
print("RMSE :", rmse_1)

print("\n비트코인 가격")
print("R2 :", r2_2)
print("MSE :", mse_2)
print("RMSE :", rmse_2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 화성 온도
# R2 : 1.0
# MSE : 2.5689600988698658e-06
# RMSE : 0.0016027975361794233

# 비트코인 가격
# R2 : 1.0
# MSE : 9.5367431640625e-07
# RMSE : 0.0009765625
# 걸린시간 :  99.43 초
