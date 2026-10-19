# 75 카피

# 다양한 사전학습 CNN 모델의 가중치를 동결하고 전이학습 기반 이미지 분류 모델의 성능을 비교

import warnings
warnings.filterwarnings("ignore")

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from tensorflow.keras.datasets import cifar10, cifar100

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

# 2. 모델
for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

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
    model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=0, validation_split=0.2, callbacks=[es, reduce_lr])

    # 4. 평가, 예측
    print("====================================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("모델명 :", model_i.name, "loss :", loss[0], "acc :", loss[1])

# ====================================
# 모델명 : vgg19 loss : 1.2172718048095703 acc : 0.578499972820282
# ====================================
# 모델명 : resnet50 loss : 1.5805431604385376 acc : 0.4438999891281128
# ====================================
# 모델명 : resnet101 loss : 1.8111798763275146 acc : 0.36309999227523804
# ====================================
# 모델명 : densenet201 loss : 1.1078417301177979 acc : 0.620199978351593
# ====================================
# 모델명 : mobilenetv2_1.00_224 loss : 1.8975979089736938 acc : 0.31119999289512634