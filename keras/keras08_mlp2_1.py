import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]])    # (3, 10)
x = x.transpose()   # 전치
# x = x.T
y = np.array([1,2,3,4,5,6,7,8,9,10])   # (10,)

print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) # input_dim은 열의 개수
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=6)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)   # loss : 0.7010117769241333
results = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값 :", results)    # [10, 1.3, 0]의 예측값 : [[4.4746437]]