import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test)) # 0 225

# 스케일링 1
# x_train = x_train/255. # 뒤에 .을 붙이면 float형으로 반환
# x_test = x_test/255.
# print(np.min(x_train), np.max(x_train)) # 0.0 1.0
# print(np.min(x_test), np.max(x_test)) # 0.0 1.0

# 스케일링 2 이미지에서 많이 쓰는 방법 -1 1 사이로도 함
x_train = (x_train - 127.5)/127.5 # 뒤에 .을 붙이면 float형으로 반환
x_test = (x_test - 127.5)/127.5
print(np.min(x_train), np.max(x_train)) # -1.0 1.0
print(np.min(x_test), np.max(x_test)) # -1.0 1.0

# 모델에 넣기 위해 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# OneHotEncoder에 넣기 위해 reshape
print(y_train) # [5 0 4 ... 5 6 8]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train)
# [[5]
#  [0]
#  [4]
#  ...
#  [5]
#  [6]
#  [8]]

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000, 10)


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1))) # (26, 26, 64) # input_shape=(height, width, channel)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # (24, 24, 32)
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) # (23, 23, 32)
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # (22, 22, 16)
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu')) # (20, 20, 16)
model.add(Flatten()) # (6400, )
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
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
print(y_pred) # [7 2 1 ... 4 5 6]
print(y_pred.shape) # (10000,)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# CPU
# oss : 0.05231226235628128
# acc : 0.9897000193595886
# accuracy_score : 0.9897
# 걸린시간 : 1705 초

# GPU
# loss : 0.04199329391121864
# acc : 0.991100013256073
# accuracy_score : 0.9911
# 걸린시간 : 141 초