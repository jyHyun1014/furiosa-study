# 39-2 카피

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],

# # 사진 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'grey')
# plt.show()

# 스케일링 1
print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test)) # 0 225
x_train = x_train/255.
x_test = x_test/255.
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

# 모델에 넣기 위해 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# OneHotEncoder에 넣기 위해 reshape
print(y_train) # [9 0 0 ... 3 0 5]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train)
# [[9]
#  [0]
#  [0]
#  ...
#  [3]
#  [0]
#  [5]]

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1))) # (26, 26, 32) # input_shape=(height, width, channel)
model.add(MaxPooling2D(pool_size=(2,2))) # (13, 13, 32)
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # (11, 11, 64)
model.add(MaxPooling2D()) # (5, 5, 64)
model.add(Dropout(0.2))
# model.add(Flatten()) # (1600, )
model.add(GlobalAveragePooling2D()) # (64,)
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # (10, )

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1,
          validation_split=0.2,
          )
end_time = time.time()

# 4. 평가, 예측
print("================ model. evaluate ====================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss :", loss[0])
print("acc :", loss[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape) # (10000, 10)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred) # [9 2 1 ... 8 1 5]
print(y_pred.shape) # (10000,)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.8789
# 걸린시간 : 51 초