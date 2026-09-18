# 44-1 복사

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time


print(np.__version__)

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1/255., # 스케일링
    # rotation_range=5, # 회전
    # width_shift_range=0.1, # 가로 이동
    # height_shift_range=0.1, # 세로 이동
    # zoom_range=0.2, # 1 ± 0.2 확대/축소
    # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    # fill_mode='nearest', # 변환 후 빈 영역 처리
)
test_datagen = ImageDataGenerator(
    rescale=1/255.,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory( # Found 160 images belonging to 2 classes.
    path_train, # 경로
    target_size=(150,150),
    batch_size=160, # 1 * (160, 100, 100, 1)
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory( # Found 120 images belonging to 2 classes
    path_test, # 경로
    target_size=(150,150),
    batch_size=120,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False, # test에서는 필요가 없다
)

# 딱 한번 변환된 이미지 데이터. 이렇게 사용하면 fit할때는 더이상 랜덤하게 변환이 안됨.
x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (160, 150, 150, 1) (160,) (120, 150, 150, 1) (120,)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu',
    input_shape=(150, 150, 1)
))

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D()) # (64,)
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # (100,)

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000000, batch_size=128,
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

y_pred = model.predict(x_test) # 0과 1사이의 실수 값으로 나옴
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred[:5])


acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 1.0
# 걸린시간 : 39 초