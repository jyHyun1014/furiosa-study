import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).transpose()
print(x.shape, y.shape)  # (10,) (10, 3)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)   # loss : 1.5309309809447003e-12
results = model.predict(np.array([10]))
print("[10]의 예측값 :", results)    # [10]의 예측값 : [[ 1.1000001e+01 -6.8545341e-07 -1.0000005e+00]]