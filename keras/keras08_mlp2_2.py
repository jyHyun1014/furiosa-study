import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(10))    # [0 1 2 3 4 5 6 7 8 9]
print(x)

x = np.array(range(1, 10))    # [1 2 3 4 5 6 7 8 9]
print(x)

x = np.array(range(1, 11))    # [1 2 3 4 5 6 7 8 9 10]
print(x)

x = np.array([range(10), range(21, 31), range(201, 211)])   # [[  0   1   2   3   4   5   6   7   8   9]
print(x)                                                    # [ 21  22  23  24  25  26  27  28  29  30]
                                                            # [201 202 203 204 205 206 207 208 209 210]]

x = np.array([range(10), range(21, 31), range(201, 211)]).T
print(x.shape)  # (10, 3)

y = np.array(range(1, 11))
print(y.shape)  # (10,)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)   # loss : 3.1402896949472847e-10
results = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 :", results)    # [10, 31, 211]의 예측값 : [[11.000002]]