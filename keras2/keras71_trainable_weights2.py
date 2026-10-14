# 71-1 카피

# 모델의 전체 가중치를 동결한 경우와 동결하지 않은 경우의 학습 결과를 비교하여 가중치 동결(Freeze)이 모델 학습 및 예측에 미치는 영향을 확인

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print(tf.__version__)

# 난수 시드 고정
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5,])
y = np.array([1,2,3,4,5,])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print("========================================================= 동결 =========================================================")
model.trainable = False
# model.trainable = True

model.summary()

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, batch_size=1, epochs=100, verbose=1)

# 4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

# 동결 O
# [[0.4565795]
#  [0.913159 ]
#  [1.3704137]
#  [1.826318 ]
#  [2.283362 ]]

# 동결 X
# [[0.99949855]
#  [1.9993956 ]
#  [3.0004833 ]
#  [3.9997163 ]
#  [4.998995  ]]