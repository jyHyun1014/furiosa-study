# 52-3 카피

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

# # 사진 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()
# print(y_train[0])

# 스케일링 1
print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test)) # 0 225
x_train = x_train/255.
x_test = x_test/255.
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 10)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D( # (32,32,32)
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu',
    input_shape=(32,32,3)
))
model.add(BatchNormalization())
model.add(Conv2D( # (32,32,32)
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # (16,16,32)
model.add(Dropout(0.2))

model.add(Conv2D( # (16,16,64)
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(Conv2D( # (16,16,64)
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # (8,8,64)
model.add(Dropout(0.3))

model.add(Conv2D( # (8,8,128)
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(Conv2D( # (8,8,128)
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # (4,4,128)
model.add(Dropout(0.4))

# model.add(Flatten()) # (2048,)
model.add(GlobalAveragePooling2D()) # (128,)
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # (10,)

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau( # val_loss가 10 epoch 동안 개선되지 않으면 Learning Rate를 절반으로 줄임
    monitor="val_loss",
    mode="auto",
    patience=10,
    verbose=1,
    factor=0.5,
)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01),
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, reduce_lr],
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
print(y_pred) # [3 8 8 ... 5 0 7]
print(y_pred.shape) # (10000,)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.8456
# 걸린시간 : 247 초

# learning_rate=0.01
# accuracy_score : 0.8391
# 걸린시간 : 253 초

# learning_rate = 0.01
# es patience=20
# reduce_lr patience=10
# accuracy_score : 0.854
# 걸린시간 : 594 초