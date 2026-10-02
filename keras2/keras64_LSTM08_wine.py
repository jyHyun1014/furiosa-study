# 53-8 카피

# 분류 tabular 데이터로 RNN 모델링하기

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
import datetime

# acc 0.95이상 만들어보기

# 1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99, stratify=y)
print(x_train.shape, y_train.shape) # (142, 13) (142, 3)
print(x_test.shape, y_test.shape) # (36, 13) (36, 3)

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# RNN 모델에 넣기 위해 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (142, 13, 1) (142,) (36, 13, 1) (36,)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(13, 1)))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(3, activation='softmax'))

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

model.compile(
    loss='sparse_categorical_crossentropy', # 정수 레이블인 경우 sparse_categorical_crossentropy
    optimizer=Adam(learning_rate=0.01), 
    metrics=['acc'],
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, reduce_lr],
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", round(result[1], 2))
# loss : 0.013210393488407135
# acc : 1.0

y_pred = model.predict(x_test)
# print(y_pred[:3])
# [[2.3223998e-04 9.5266491e-01 4.7102865e-02]
#  [2.3241100e-01 5.3238034e-01 2.3520868e-01]
#  [3.8398942e-01 4.1711840e-01 1.9889218e-01]
#  ...

y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
print(y_pred) # [1 1 2 1 1 1 2 1 0 0 0 0 2 0 2 0 1 2 1 1 1 1 2 2 1 1 2 0 1 0 2 1 1 0 0 0]

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score) # acc_score : 0.9118833333333334
print("걸린시간 :", round(end_time - start_time), "초")

# acc_score : 0.9166666666666666
# 걸린시간 : 3 초