# 73-14 카피

# ImageNet 사전 학습된 VGG16을 CIFAR-10 이미지 분류에 적용하고, 가중치 동결 여부(trainable=True/False)에 따른 분류 정확도와 학습 시간을 비교하는 전이학습

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [
    VGG19(include_top=False, input_shape=(32, 32, 3),),
    ResNet50(include_top=False, input_shape=(32, 32, 3),),
    ResNet101(include_top=False, input_shape=(32, 32, 3),),
    DenseNet201(include_top=False, input_shape=(32, 32, 3),),
    # InceptionV3(include_top=False, input_shape=(32, 32, 3),), 
    # InceptionResNetV2(include_top=False, input_shape=(32, 32, 3),),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3),),
    # EfficientNetB0(include_top=False, input_shape=(32, 32, 3),),
]

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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
for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(100, activation='softmax'))

    # 3. 컴파일, 훈련
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["acc"])
    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="auto",
        patience=5,
        verbose=0,
        factor=0.5,
    )
    model.fit(x_train, y_train, epochs=50, batch_size=512, verbose=0, validation_split=0.2, callbacks=[es, reduce_lr])

    # 4. 평가, 예측
    print("====================================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("모델명 :", model_i.name, "loss :", loss[0], "acc :", loss[1])