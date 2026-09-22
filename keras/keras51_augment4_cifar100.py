# 43-4 카피

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) # (array([0, 1 , ..., 99], dtype=uint8), array([500, 500, ... , 500],

# # 사진 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()
# print(y_train[0])

# 이미지 증강 설정
datagen = ImageDataGenerator(
    # rescale=1/255., # 스케일링  # 픽셀값을 0~255 → 0~1로 변환
    rotation_range=5, # 최대 ±5° 회전
    width_shift_range=0.1, # 가로 이동
    height_shift_range=0.1, # 세로 이동
    zoom_range=0.1, # 1 ± 0.1 확대/축소
    # shear_range=0.7, # 이미지 기울이기(전단 변환)
    # horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    fill_mode='nearest', # 변환 후 빈 영역을 주변 픽셀로 채움
)


###################################### 증강할 데이터 40,000장 선택 ######################################
augment_size = 40000 # 6만개 데이터 중 변환할 데이터 개수

print(x_train.shape) # (50000, 32, 32, 3)
print(x_train.shape[0]) # 50000

# 50,000장의 train 데이터에서 40,000개의 인덱스를 랜덤으로 선택
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) #[0, 50000) 범위 무작위 정수를 4만개 선택

print(randidx) # [18478  6912 18998 ...  7031 45883 37106] # 랜덤으로 선택된 인덱스
print(type(randidx)) # <class 'numpy.ndarray'>
print(randidx.shape) # (40000,) # 리스트는 .shape가 안되는데 randidx는 넘파이 배열 객체임
print(len(randidx)) # 40000
print(np.min(randidx), np.max(randidx)) # 0 ~ 49999 범위

# 선택한 이미지와 label 복사
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape) # (40000, 32, 32, 3) (40000, 1)


###################################### 이미지 변환(증강) ######################################
# 선택한 40,000장에 설정한 변환을 적용하여 새로운 이미지 생성
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,  # 40,000장을 한 번에 생성
    shuffle=False,

).next()[0] # (이미지, label) 중 이미지 데이터만 가져옴

print(x_augmented.shape) # (40000, 32, 32, 3)


###################################### 원본 데이터 + 증강 데이터 결합  ######################################
# 원본 50,000장 + 증강 40,000장 = 총 90,000장
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (90000, 32, 32, 3) (90000, 1)

# 클래스별 데이터 개수 확인
print(np.unique(y_train, return_counts=True))


###################################### 데이터 전처리 ######################################
# 스케일링
print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test)) # 0 225
x_train = x_train/255.
x_test = x_test/255.
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

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

model.add(Flatten()) # (2048,)
# model.add(GlobalAveragePooling2D()) # (128,)
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax')) # (100,)

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

path = "./_save/keras51/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k51_03_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp],
          )
end_time = time.time()

# 4. 평가, 예측
print("================ model. evaluate ====================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss :", loss[0])
print("acc :", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.6073
# 걸린시간 : 1151 초