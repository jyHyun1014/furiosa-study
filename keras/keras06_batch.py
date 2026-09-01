from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=3) # 데이터 3개씩 훈련시킴
# 데이터가 많아질수록 batch가 빛을 발한다
# batch size를 명시하지 않았을때, batch_size=32

# 학습할 데이터가 6개 있을때,
# batch_size=1 => (1, 1, 1, 1, 1, 1)개씩 학습함
# batch_size=2 => (2, 2, 2)개씩 학습함
# batch_size=3 => (3, 3)개씩 학습함
# batch_size=4 => (4, 2)개씩 학습함
# batch_size=5 => (5, 1)개씩 학습함
# batch_size=5 => 6개씩(한번에) 학습함

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)
# result = model.predict(np.array([1,2,3,4,5,6]))
# print("예측값 :", result)