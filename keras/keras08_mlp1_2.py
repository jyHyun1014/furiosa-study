import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])    # (2, 5)
# x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]])  # (5, 2)
x = x.transpose()   # 전치
# x = x.T
y = np.array([1,2,3,4,5])   # (5,)

print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) # input_dim은 열의 개수 # 행무시 열우선
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)   # loss : 0.7010117769241333
results = model.predict(np.array([[6, 11]]))
print("[6, 11]의 예측값 :", results)    # [6, 11]의 예측값 : [[4.4746437]]