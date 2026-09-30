# 55-2의 데이터만 가져옴

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True))
# 모든 timestep의 hidden state를 다음 RNN layer에 전달하기 위해 3차원으로 출력
# return_sequences=False(기본값)이면 마지막 timestep의 hidden state만 출력하여 2차원이 됨

model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5))
# 마지막 LSTM: 마지막 timestep의 hidden state만 출력
# return_sequences=False(기본값)이므로 2차원 출력 → 다음 Dense layer로 전달

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 3. 컴파일
es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="loss",
    patience=30,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
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
# loss : 0.011141275055706501
# 1/1 [==============================] - 0s 482ms/step
# [50,60,70]의 결과 : [[74.09846]]