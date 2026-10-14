# 72-1 카피

# ImageNet 사전 학습된 VGG16을 CIFAR-10 이미지 분류에 적용하고, 가중치 동결 여부(trainable=True/False)에 따른 분류 정확도와 학습 시간을 비교하는 전이학습

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

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

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# 2. 모델
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)

vgg16.trainable = True # 모든 가중치 학습
# vgg16.trainable = False # VGG16의 가중치 학습 중지
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="auto",
    patience=5,
    verbose=1,
    factor=0.5,
)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01),
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=512,
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
print(y_pred) # [3 8 8 ... 5 0 7]
print(y_pred.shape) # (10000,)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# 실습 # 
# 비교할거
# 1. 이전에 본인이 한 최상의 결과
# 63-3
# accuracy_score : 0.854
# 걸린시간 : 594 초

# 2. 가중치를 동결하고 훈련시켰을때, trainable=False
# accuracy_score : 0.5866
# 걸린시간 : 154 초

# 3. 가중치를 동결하지 않고 훈련시켰을때, trainable=True 
# accuracy_score : 0.2614
# 걸린시간 : 154 초

# 4. 시간까지 비교


# 결론: CIFAR-10은 이미지 크기가 32 × 32로 작기 때문에 VGG16의 5개 MaxPooling Layer를 거치면 특징 맵이 1 × 1까지 축소되어 공간 정보가 손실되고, 
# 이로 인해 분류 성능이 저하될 수 있음.
