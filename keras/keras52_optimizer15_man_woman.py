# 1. 데이터
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import datetime

# np array 불러오기
np_path = "./_save/keras46/"
x_train = np.load(np_path + "keras46_03_x_train.npy")
y_train = np.load(np_path + "keras46_03_y_train.npy")
x_test = np.load(np_path + "keras46_03_x_test.npy")
y_test = np.load(np_path + "keras46_03_y_test.npy")

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (21733, 100, 100, 3) (21733,) (5434, 100, 100, 3) (5434,)


# 2. 모델구성
model = Sequential()
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu',
    input_shape=(100, 100, 3)
))

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='relu'))
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

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01),
              metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000000, batch_size=5,
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
y_pred = np.round(y_pred)
print(y_pred[:5])

acc_score = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc_score)
print("걸린시간 :", round(end_time - start_time), "초")

# accuracy_score : 0.8638203901361796
# 걸린시간 : 3126 초

# learning_rate=0.01
# accuracy_score : 0.728928965771071
# 걸린시간 : 228 초