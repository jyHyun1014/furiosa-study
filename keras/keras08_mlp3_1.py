import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]).T
print(x.shape)  # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]).T
print(y.shape)  # (10, 2)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=3)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)   # loss : 0.004692344460636377
results = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 :", results)    # [10, 31, 211]의 예측값 : [[10.946167   0.1835128]]