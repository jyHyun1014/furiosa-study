# 35-7 카피

# Santander Customer Transaction Prediction
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time
import datetime

# 1. 데이터
path = "./_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

print(train_csv.shape) # (200000, 201)
print(test_csv.shape) # (200000, 200)
print(submission_csv.shape) # (200000, 2)

print(train_csv.isna().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 201) (200000,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
test_csv = scaler.transform(test_csv)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='swish', input_dim=200))
model.add(Dropout(0.4))
model.add(Dense(10, activation='swish'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', # 가중치 업데이트에 사용
              optimizer=Adam(learning_rate=0.01),
            #   metrics=['accuracy'],
              metrics=['acc'], # 성능 확인용
              )

start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=100,
    batch_size=100,
    validation_data=(x_val, y_val),
    )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_val, y_val)
print("===========================")
print("loss :", loss[0])
print("acc :", round(loss[1], 4))
# loss : 0.23688732087612152
# acc : 0.9119
print("===========================")

y_pred = model.predict(x_val)
y_pred = np.round(y_pred)
print(y_pred[:5])

acc_score = accuracy_score(y_val, y_pred)
print("acc_score :", acc_score) # acc_score : 0.9118833333333334

print("걸린시간 : ", round(end_time - start_time, 2), "초") # 걸린시간 :  459.03 초

# CPU
# acc_score : 0.91265
# 걸린시간 :  182.56 초

# GPU
# acc_score : 0.9121
# 걸린시간 :  384.43 초

# learning_rate=0.01
# acc_score : 0.899975
# 걸린시간 :  437.46 초