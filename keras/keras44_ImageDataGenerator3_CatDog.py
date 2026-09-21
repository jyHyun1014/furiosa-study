# https://www.kaggle.com/datasets/tongpython/cat-and-dog

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time
import datetime


path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=5, # 회전
    width_shift_range=0.1, # 가로 이동
    height_shift_range=0.1, # 세로 이동
    # zoom_range=0.2, # 1 ± 0.2 확대/축소
    # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    fill_mode='nearest', # 변환 후 빈 영역 처리
    validation_split=0.2,
)
test_datagen = ImageDataGenerator(
    rescale=1/255.,
)

# 학습데이터
xy_train = train_datagen.flow_from_directory( # Found 8005 images belonging to 2 classes.
    path_train, # 경로
    target_size=(150,150),
    batch_size=32, #
    class_mode='binary', # 이진분류
    color_mode='rgb', # 컬러
    shuffle=True,
    subset='training',
)

# 검증 데이터
xy_val = train_datagen.flow_from_directory(
    path_train, 
    target_size=(150,150), 
    batch_size=32, 
    class_mode='binary', 
    color_mode='rgb', 
    shuffle=False, 
    subset='validation',
)

# 테스트 데이터
xy_test = test_datagen.flow_from_directory( # Found 2023 images belonging to 2 classes.
    path_test, # 경로
    target_size=(150,150),
    batch_size=32,
    class_mode='binary', # 이진분류
    color_mode='rgb', # 컬러
    shuffle=False, # 예측값과 실제 label의 순서 일치
)

print('train:', xy_train.samples) # 6404
print('validation:', xy_val.samples) # 1601
print('test:', xy_test.samples) # 2023

print('class_indices :', xy_train.class_indices) # class_indices : {'cats': 0, 'dogs': 1}

# 2. 모델구성
model = Sequential()
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu',
    input_shape=(150,150,3)
))
model.add(BatchNormalization())
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
    verbose=1,
)

path = "./_save/keras44/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k44_03_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(
    xy_train, 
    epochs=1000000,
    # batch_size=10, # flow_from_directory에서 32장씩 batch를 만들어서 전달하고 있기 때문에 사용하지 않음
    verbose=1,
    validation_data=xy_val,
    callbacks=[es, mcp],
)
end_time = time.time()

# model = load_model("./_save/keras44/k44_03_0918_1611_0026-0.2813.keras") # 모델 불러오기

# 4. 평가, 예측
print("================ model. evaluate ====================")
loss = model.evaluate(xy_test, verbose=1)
print("loss :", loss[0])
print("acc :", loss[1])
# loss : 0.20660245418548584
# acc : 0.9149777293205261

y_pred = model.predict(xy_test) # 0과 1사이의 실수 값으로 나옴
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred[:5])

y_test = xy_test.classes
acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.9149777558082056
# 걸린시간 : 1395 초