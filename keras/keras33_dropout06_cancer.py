# 31-6 카피

# 이진분류

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR) # 데이터 설명

x = datasets.data
# x = datasets['data']
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)
# print(type(x)) # <class 'numpy.ndarray'>

# print(y)
# print(type(y)) # <class 'numpy.ndarray'>
# print(np.unique(y)) # [0 1]
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
# print(pd.DataFrame(y).value_counts())
# # 1    357
# # 0    212
# print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([170, 285]))
print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([42, 72]))
print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape) # (114, 30) (114,)

# 스케일링
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
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
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

path = "./_save/keras33/"
date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, "k33_06_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)

model.compile(loss='binary_crossentropy', # 가중치 업데이트에 사용
              optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['acc'], # 성능 확인용
              )


start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=10000, 
    batch_size=5, 
    validation_split=0.2, 
    callbacks=[es, mcp]
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

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("Cancer Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()

"""
##########################################
MaxAbsScaler 적용
loss : 0.06283757835626602
acc_score : 0.9824561403508771
걸린시간 :  23.92 초
###########################################
"""

"""
##########################################
RobustScaler 적용
loss : 0.21371710300445557
acc_score : 0.9385964912280702
걸린시간 :  9.46 초
###########################################
"""

"""
##########################################
RobustScaler 적용 (ModelCheckpoint)
loss : 0.09713049232959747
acc_score : 0.9736842105263158
걸린시간 :  7.96 초
###########################################
"""