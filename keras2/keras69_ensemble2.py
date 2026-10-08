# 서로 다른 세개의 입력 데이터를 각각 신경망으로 처리한 후 결합하여 하나의 연속적인 값을 예측하는 다중 입력 회귀 모델

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

x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).transpose() # (100, 4)

# 예측 대상: 화성의 화씨 온도
y = np.array(range(3001, 3101))

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y, test_size=0.2, random_state=42)

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델2
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

# 2-3. 모델3
input31 = Input(shape=(4,))
dense31 = Dense(50, name='han31')(input31)
dense32 = Dense(40, name='han32')(dense31)
dense33 = Dense(30, name='han33')(dense32)
dense34 = Dense(20, name='han34')(dense33)
output31 = Dense(3, name='han35')(dense34)
# model3 = Model(inputs=input21, outputs=output21)

# 2-4 모델 결합
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output21, output31], name="mg1") # 첫 번째 모델의 출력 5개와 두 번째 모델의 출력 3개와 세 번째 모델의 출력 3개를 연결
merge1 = Concatenate(name="mg1")([output1, output21, output31]) # 결합된 데이터를 추가 신경망에 통과시켜 최종 예측값 생성
merge2 = Dense(10, name="mg2")(merge1)
merge3 = Dense(5, name="mg3")(merge2)
last_output = Dense(1, name="last")(merge3)

model = Model(inputs=[input1, input21, input31], outputs=last_output)

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
model.fit([x1_train, x2_train, x3_train], y_train, 
          epochs=10000, 
          batch_size=1,
          validation_split=0.2,
          callbacks=[es, reduce_lr]
          )
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
print("=====================================")
result = model.evaluate([x1_test, x2_test, x3_test], y_test)
print("loss :", result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T
y_pred = model.predict([x1_pred, x2_pred, x3_pred])

y_pred = model.predict([x1_test, x2_test, x3_test])
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# R2 : 0.9999996423721313
# MSE : 0.0003047764184884727
# RMSE : 0.01745784655213356
# 걸린시간 :  7.54 초

