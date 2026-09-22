# 50-2 카피

# Fashion-MNIST의 기존 60,000장 중 40,000장을 랜덤으로 골라 이미지 변환(증강)한 뒤, 
# 원본 60,000장과 합쳐서 100,000장의 학습 데이터를 만드는 코드


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time
import datetime

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 클래스별 데이터 개수 확인
print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],

# # 사진 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'grey')
# plt.show()

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

print(x_train.shape) # (60000, 28, 28)
print(x_train.shape[0]) # 60000

# 60,000장의 train 데이터에서 40,000개의 인덱스를 랜덤으로 선택
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) #[0, 60000) 범위 무작위 정수를 4만개 선택

print(randidx) # [  610 42013 22012 ... 53732  4690  2908] # 랜덤으로 선택된 인덱스
print(type(randidx)) # <class 'numpy.ndarray'>
print(randidx.shape) # (40000,) # 리스트는 .shape가 안되는데 randidx는 넘파이 배열 객체임
print(len(randidx)) # 40000
print(np.min(randidx), np.max(randidx)) # 0 ~ 59999 범위

# 선택한 이미지와 label 복사
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# CNN 입력 형태에 맞게 채널 차원 추가
# (40000, 28, 28) → (40000, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    1
)
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28, 1) (40000,)


###################################### 이미지 변환(증강) ######################################
# 선택한 40,000장에 설정한 변환을 적용하여 새로운 이미지 생성
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,  # 40,000장을 한 번에 생성
    shuffle=False,

).next()[0] # (이미지, label) 중 이미지 데이터만 가져옴

print(x_augmented.shape) # (40000, 28, 28, 1)


###################################### 원본 train/test 데이터도 CNN 입력 형태로 변환 ######################################
x_train = x_train.reshape(60000, 28, 28, 1) # (60000, 28, 28) → (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) # (10000, 28, 28) → (10000, 28, 28, 1)


###################################### 원본 데이터 + 증강 데이터 결합  ######################################
# 원본 60,000장 + 증강 40,000장 = 총 100,000장
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# 클래스별 데이터 개수 확인
print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([ 9939, 10131,  9945,  9907,  9937, 10025,  9963, 10107, 10032, 10014], dtype=int64))

###################################### 데이터 전처리 ######################################

# 스케일링
print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test)) # 0 225
x_train = x_train/255.
x_test = x_test/255.
print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

# OneHotEncoder에 넣기 위해 reshape
print(y_train) # [9 0 0 ... 3 0 5]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000, 10)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000, 10)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu', input_shape=(28, 28, 1))) # input_shape=(height, width, channel)
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # (10, )

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
filepath = "".join([path, "k51_01_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32,
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
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.922
# 걸린시간 : 292 초