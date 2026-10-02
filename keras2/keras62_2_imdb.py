# IMDB 영화 리뷰 텍스트 데이터셋 이진분류

from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000
)

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape) # (25000,) (25000,)

print(x_train)
print(min(min(x) for x in x_train)) # 1
print(max(max(x) for x in x_train)) # 999

print(y_train) # [1 0 0 ... 0 1 0]
print(np.unique(y_train)) # [0 1]

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'numpy.ndarray'>

############ 영화 리뷰 길이 분석 ############
print('영화 리뷰의 최대 길이 :', max(len(sample) for sample in x_train)) # 영화 리뷰의 최대 길이 : 2494
print('영화 리뷰의 최소 길이 :', min(len(sample) for sample in x_train)) # 영화 리뷰의 최소 길이 : 11
print('영화 리뷰의 평균 길이 :', sum(map(len, x_train))/len(x_train)) # 영화 리뷰의 평균 길이 : 238.71364

# plt.hist([len(sample) for sample in x_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

# padding
x_train = pad_sequences(x_train, 
                         padding="pre", # 앞쪽을 0으로 채움 # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                         maxlen=500, # 모든 문장을 토큰 길이로 맞춤 (지정하지 않으면 가장 긴 시퀀스 기준)
                         truncating='post' # 문장이 길면 뒤쪽 토큰부터 제거
                         )
print(x_train)
print(x_train.shape) # (25000, 500)

x_test = pad_sequences(x_test, 
                         padding="pre", # 앞쪽을 0으로 채움 # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                         maxlen=500, # 모든 문장을 토큰 길이로 맞춤 (지정하지 않으면 가장 긴 시퀀스 기준)
                         truncating='post' # 문장이 길면 뒤쪽 토큰부터 제거
                         )
print(x_test)
print(x_test.shape) # (25000, 500)


# 2. 모델 구성
# 다양한 모델로 구현해보자

# Embedding + LSTM/GRU
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100))
model.add(LSTM(100))
model.add(Dense(10))
model.add(Dense(1, activation="sigmoid"))

# Embedding + Bidirectional
# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100))
# model.add(Bidirectional(LSTM(10)))
# model.add(Dense(10))
# model.add(Dense(1, activation="sigmoid"))

# Embedding + Flatten + DNN
# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100, input_length=500))
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Dense(1, activation="sigmoid"))

# Embedding + RNN + Flatten + DNN
# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100, input_length=500))
# model.add(LSTM(100, return_sequences=True))
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Dense(1, activation="sigmoid"))

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)
model.compile(
    loss="binary_crossentropy", 
    optimizer="adam", 
    metrics=["acc"]
)
start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=1,
    batch_size=128, 
    verbose=1,
    validation_split=0.2,
    shuffle=True,
    callbacks=[es],
)
end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("===========================")
print("loss :", loss[0]) #
print("acc :", round(loss[1], 4)) # acc 0.6 목표
print("===========================")

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
print(y_pred)

# print(classification_report(y_test, y_pred))
print("걸린시간 :", round(end_time - start_time), "초")

# ===========================
# loss : 0.3573330342769623
# acc : 0.85
# ===========================
# 걸린시간 : 8 초