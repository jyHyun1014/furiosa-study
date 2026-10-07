# 55-2 카피

# 시계열 데이터를 Conv1D로 모델링

import numpy as np
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) # 80 예측하기

x = x.reshape(-1, 3, 1)
print(x.shape) # (13, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1))) # (None, 2, 10)
model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 3. 컴파일
es = EarlyStopping(
    monitor="loss",
    patience=30,
    verbose=1,
    mode="auto",
    # restore_best_weights=True,
)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, 
          epochs=1000000,
          batch_size=32,
          verbose=1,
          callbacks=[es],
          )

# 4. 평가, 예측
results= model.evaluate(x, y)
print("loss :", results)

x_predict = x_predict.reshape(-1,3,1)
y_predict = model.predict(x_predict)

print("[50,60,70]의 결과 :", y_predict)


# Epoch 579: early stopping
# 1/1 [==============================] - 0s 142ms/step - loss: 3.5063e-04
# loss : 0.00035063058021478355
# 1/1 [==============================] - 0s 65ms/step
# [50,60,70]의 결과 : [[79.98962]]