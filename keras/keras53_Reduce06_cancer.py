# 52-6 카피

# 이진분류

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import datetime

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR) # 데이터 설명

x = datasets.data
# x = datasets['data']
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([170, 285]))
print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([42, 72]))
print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape) # (114, 30) (114,)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dropout(0.4))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 출력층 무조건 sigmoid

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping( # 20 epoch 동안 개선되지 않으면 학습을 종료
    monitor="val_loss",
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau( # val_loss가 10 epoch 동안 개선되지 않으면 Learning Rate를 절반으로 줄임
    monitor="val_loss",
    mode="auto",
    patience=10,
    verbose=1,
    factor=0.5,
)

model.compile(loss='binary_crossentropy', # 가중치 업데이트에 사용
              optimizer=Adam(learning_rate=0.01),
            #   metrics=['accuracy'],
              metrics=['acc'], # 성능 확인용
              )


start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=5, 
    validation_split=0.2,
    callbacks=[es, reduce_lr],
    )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("===========================")
print("loss :", loss[0]) #
print("acc :", round(loss[1], 4))
print("===========================")
# loss : 0.1282176822423935
# acc : 0.9737

y_pred = model.predict(x_test) # 0과 1사이의 실수 값으로 나옴
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred[:5])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")
# acc_score : 0.9736842105263158
# 걸린시간 :  9.55 초

# CPU
# acc_score : 0.956140350877193
# 걸린시간 :  13.25 초

# GPU
# acc_score : 0.9473684210526315
# 걸린시간 :  20.69 초

# learning_rate=0.01
# acc_score : 0.9649122807017544
# 걸린시간 :  20.12 초

# learning_rate = 0.01
# es patience=20
# reduce_lr patience=10
# acc_score : 0.9385964912280702
# 걸린시간 :  8.58 초