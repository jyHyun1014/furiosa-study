# 여자 데이터만 증강해서 성능 올리기

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import datetime

# 1. 데이터

# np array 불러오기
np_path = "./_save/keras46/"
x_train = np.load(np_path + "keras46_03_x_train.npy")
y_train = np.load(np_path + "keras46_03_y_train.npy")
x_test = np.load(np_path + "keras46_03_x_test.npy")
y_test = np.load(np_path + "keras46_03_y_test.npy")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (21733, 100, 100, 3) (21733,) (5434, 100, 100, 3) (5434,)

# 클래스별 데이터 개수 확인
print(np.unique(y_train, return_counts=True)) # (array([0., 1.], dtype=float32), array([14142,  7591], dtype=int64))
# 남자 데이터 14142개, 여자 데이터 7591개로 데이터 불균형이 있으므로 여자 데이터만 2배로 증강할 예정

# 여자 학습 데이터만 추출
x_train_woman = x_train[np.where(y_train > 0)]
y_train_woman = y_train[np.where(y_train > 0)]
print(x_train_woman.shape, y_train_woman.shape) # (7591, 100, 100, 3) (7591,)

###################################### 이미지 변환(증강) ######################################
# 이미지 증강 설정
datagen = ImageDataGenerator(
    # rescale=1/255., # 스케일링
    # rotation_range=5, # 회전
    # width_shift_range=0.1, # 가로 이동
    # height_shift_range=0.1, # 세로 이동
    # zoom_range=0.1, # 1 ± 0.1 확대/축소
    # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    fill_mode='nearest', # 변환 후 빈 영역 처리
)

# 선택한 40,000장에 설정한 변환을 적용하여 새로운 이미지 생성
x_train_woman = datagen.flow(
    x_train_woman, y_train_woman,
    batch_size=x_train_woman.shape[0],  # 40,000장을 한 번에 생성
    shuffle=False,

).next()[0] # (이미지, label) 중 이미지 데이터만 가져옴

print(x_train_woman.shape) # (7591, 100, 100, 3)


###################################### 원본 데이터 + 증강 데이터 결합  ######################################
# 원본 21733장 + 증강 7591장 = 총 29324장
x_train = np.concatenate((x_train, x_train_woman))
y_train = np.concatenate((y_train, y_train_woman))
print(x_train.shape, y_train.shape) # (29324, 100, 100, 3) (29324,)

# 클래스별 데이터 개수 확인
print(np.unique(y_train, return_counts=True)) # (array([0., 1.], dtype=float32), array([14142, 15182], dtype=int64))


###################################### 데이터 전처리 ######################################
# 이미 스케일링 되어있으므로 스케일링은 패스
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0


# 2. 모델구성
model = Sequential()
model.add(Conv2D(
    filters=64,
    kernel_size=(5,5),
    padding='same',
    activation='relu',
    input_shape=(100, 100, 3)
))

model.add(Conv2D(
    filters=64,
    kernel_size=(5,5),
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(
    filters=16,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(Conv2D(
    filters=16,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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
filepath = "".join([path, "k51_05_", date, filename]) # 모델 저장 경로

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
model.fit(x_train, y_train, epochs=1000000, batch_size=100,
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
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.9100110415899889
# 걸린시간 : 457 초
