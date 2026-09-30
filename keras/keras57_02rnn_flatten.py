# 57-1 카피
# flatten 적용해서 57-1번과 성능 비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
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
model.add(LSTM(units=32, input_shape=(3, 1), return_sequences=True))
# 모든 timestep의 hidden state를 다음 RNN layer에 전달하기 위해 3차원으로 출력
# return_sequences=False(기본값)이면 마지막 timestep의 hidden state만 출력하여 2차원이 됨

model.add(LSTM(32, return_sequences=True))
model.add(LSTM(64, return_sequences=True))

model.add(Flatten())
# LSTM의 3차원 출력 데이터를 2차원으로 펼침. Dense layer에 전달하기 위해
# (batch_size, 3, 5) → (batch_size, 15)

model.add(Dense(64, activation='relu'))
model.add(Dense(32))
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
# Epoch 3899: early stopping
# 1/1 [==============================] - 1s 575ms/step - loss: 0.0046
# loss : 0.0046447450295090675
# 1/1 [==============================] - 0s 466ms/step
# [50,60,70]의 결과 : [[78.57459]]