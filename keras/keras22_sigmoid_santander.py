# Santander Customer Transaction Prediction
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='swish', input_dim=200))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', # 가중치 업데이트에 사용
              optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['acc'], # 성능 확인용
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)
start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=1000000000,
    batch_size=100,
    validation_data=(x_val, y_val),
    callbacks=[es]
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

# ####################### 제출용 #################
y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
submission_csv['target'] = y_submit
submission_csv.loc[submission_csv['target'] >= 0.4, 'target'] = 1
submission_csv.loc[submission_csv['target'] < 0.4, 'target'] = 0

# print(submission)
# print(submission.shape)

submission_csv.to_csv(path + "submit/" + "submit_0908_1732.csv", index=False)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("Santander Customer Transaction Prediction Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()


'''
# 1차 시도 submit_0908_1649
random_stat = 42
test_size=0.3
epochs = 1000000000
batch_size = 100

monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
loss : 0.23688732087612152
acc : 0.9119
# 캐글 점수
ROC-curve : 0.62695
'''

'''
# 2차 시도 submit_0908_1717
random_stat = 42
test_size=0.2
epochs = 1000000000
batch_size = 100

monitor='val_loss',
mode='min',
patience=30,
restore_best_weights=True,

# 결과
loss : 0.2377249300479889
acc : 0.9116
# 캐글 점수
ROC-curve : 0.62776
'''

'''
# 4차 시도 submit_0908_1732
random_stat = 42
test_size=0.2
activation='swish'
epochs = 1000000000
batch_size = 100

monitor='val_loss',
mode='min',
patience=30,
restore_best_weights=True,

임계값을 0.4로 낮춤

# 결과
loss : 0.2381867915391922
acc : 0.912
# 캐글 점수
ROC-curve : 0.65535
'''