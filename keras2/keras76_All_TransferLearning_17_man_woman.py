# 46-3 카피
# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

# 다양한 사전학습 CNN 모델의 가중치를 동결하고 전이학습 기반 이미지 분류 모델의 성능을 비교

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.applications import VGG16
import time
import datetime
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# # 1. 데이터
# path_data = './_data/image/man_woman/'
# datagen = ImageDataGenerator(
#     rescale=1/255.,
#     # rotation_range=5, # 회전
#     # width_shift_range=0.1, # 가로 이동
#     # height_shift_range=0.1, # 세로 이동
#     # zoom_range=0.2, # 1 ± 0.2 확대/축소
#     # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
#     # horizontal_flip = True, # 좌우 반전
#     # vertical_flip=True, # 상하 반전
#     # fill_mode='nearest', # 변환 후 빈 영역 처리
#     # validation_split=0.2,
# )

# xy = datagen.flow_from_directory( # Found 27167 images belonging to 2 classes.
#     path_data, # 경로
#     target_size=(100,100),
#     batch_size=10000000,
#     class_mode='binary', # 이진분류
#     color_mode='rgb', # 컬러
#     shuffle=False,
# )
# print('class_indices :', xy.class_indices) # class_indices : {'man': 0, 'woman': 1}

# x = xy[0][0]
# y = xy[0][1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (21733, 100, 100, 3) (21733,) (5434, 100, 100, 3) (5434,)

# # np array 저장
# np_path = "./_save/keras46/"
# np.save(np_path + "keras46_03_x_train.npy", arr=x_train)
# np.save(np_path + "keras46_03_y_train.npy", arr=y_train)
# np.save(np_path + "keras46_03_x_test.npy", arr=x_test)
# np.save(np_path + "keras46_03_y_test.npy", arr=y_test)

# np array 불러오기
np_path = "./_save/keras46/"
x_train = np.load(np_path + "keras46_03_x_train.npy")
y_train = np.load(np_path + "keras46_03_y_train.npy")
x_test = np.load(np_path + "keras46_03_x_test.npy")
y_test = np.load(np_path + "keras46_03_y_test.npy")

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (21733, 100, 100, 3) (21733,) (5434, 100, 100, 3) (5434,)

# 2. 모델구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3),
)
vgg16.trainable = True # VGG16의 가중치 학습 중지

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=3,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
print("================ model. evaluate ====================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss :", loss[0])
print("acc :", loss[1])

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred[:5])

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.9418476260581524
# 걸린시간 : 410 초

