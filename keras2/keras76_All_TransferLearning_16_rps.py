# 47-2 카피

# 다양한 사전학습 CNN 모델의 가중치를 동결하고 전이학습 기반 이미지 분류 모델의 성능을 비교

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
    VGG19(include_top=False, input_shape=(150, 150, 3),),
    ResNet50(include_top=False, input_shape=(150, 150, 3),),
    ResNet101(include_top=False, input_shape=(150, 150, 3),),
    DenseNet201(include_top=False, input_shape=(150, 150, 3),),
    # InceptionV3(include_top=False, input_shape=(150, 150, 3),), 
    # InceptionResNetV2(include_top=False, input_shape=(150, 150, 3),),
    MobileNetV2(include_top=False, input_shape=(150, 150, 3),),
    # EfficientNetB0(include_top=False, input_shape=(150, 150, 3),),
]

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.applications import VGG16
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
# path = './_data/image/rps/'

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
# )

# xy = datagen.flow_from_directory( # Found 2048 images belonging to 3 classes.
#     path, # 경로
#     target_size=(150,150),
#     batch_size=10000000,
#     class_mode='categorical',
#     color_mode='rgb',
#     shuffle=False,
# )

# x = xy[0][0]
# y = xy[0][1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (1638, 150, 150, 3) (1638, 3) (410, 150, 150, 3) (410, 3)

# # np array 저장
np_path = "./_save/keras46/"
# np.save(np_path + "keras46_02_x_train.npy", arr=x_train)
# np.save(np_path + "keras46_02_y_train.npy", arr=y_train)
# np.save(np_path + "keras46_02_x_test.npy", arr=x_test)
# np.save(np_path + "keras46_02_y_test.npy", arr=y_test)

# np array 불러오기
x_train = np.load(np_path + "keras46_02_x_train.npy")
y_train = np.load(np_path + "keras46_02_y_train.npy")
x_test = np.load(np_path + "keras46_02_x_test.npy")
y_test = np.load(np_path + "keras46_02_y_test.npy")

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (1638, 150, 150, 3) (1638, 3) (410, 150, 150, 3) (410, 3)

# 2. 모델
for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(3, activation='softmax'))

    # 3. 컴파일, 훈련
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["acc"])
    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="auto",
        patience=2,
        verbose=0,
        factor=0.5,
    )
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[es, reduce_lr])

    # 4. 평가, 예측
    print("====================================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("모델명 :", model_i.name, "loss :", loss[0], "acc :", loss[1])


# 2. 모델구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
)
vgg16.trainable = False # VGG16의 가중치 학습 중지

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=3,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=64,
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
y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
y_test = np.argmax(y_test, axis=1) # 가장 확률이 높은 인덱스 선택
print(y_pred[:5])

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 1.0
# 걸린시간 : 142 초
