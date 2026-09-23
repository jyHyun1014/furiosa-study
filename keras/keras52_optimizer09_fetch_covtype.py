# 35-9 카피

# 다중분류

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time
import datetime

# acc 0.93

# 1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))


print(y) # [5 5 2 ... 3 3 3]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y) # [4 4 1 ... 2 2 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99, stratify=y)
print(x_train.shape, y_train.shape) # (464809, 54) (464809,)
print(x_test.shape, y_test.shape) # (116203, 54) (116203,)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(20, activation='swish', input_dim=54))
model.add(Dropout(0.4))
model.add(Dense(20, activation='swish'))
model.add(Dropout(0.4))
model.add(Dense(15, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(
    loss='sparse_categorical_crossentropy', # 정수 레이블인 경우 sparse_categorical_crossentropy
    optimizer=Adam(learning_rate=0.01), 
    metrics=['acc'],
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000,
          verbose=1,
          validation_split=0.2,
          )
end_time = time.time()


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", round(result[1], 2))
# loss : 0.36120179295539856
# acc : 0.85

y_pred = model.predict(x_test)
# print(y_pred)
# [[4.2225206e-01 5.4432935e-01 8.1924336e-05 ... 2.0792518e-02
#   7.6491653e-04 1.1779276e-02]
#  ...

y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
y_pred = le.inverse_transform(y_pred)
y_test = le.inverse_transform(y_test)
print(y_pred) # [1 2 2 ... 1 2 2]

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score) # acc_score : 0.9118833333333334
print("걸린시간 :", round(end_time - start_time), "초")

# CPU
# acc_score : 0.7432940629759989
# 걸린시간 : 83 초

# GPU
# acc_score : 0.7433887249038321
# 걸린시간 : 100 초

# learning_rate=0.01
# acc_score : 0.753870382003907
# 걸린시간 : 92 초