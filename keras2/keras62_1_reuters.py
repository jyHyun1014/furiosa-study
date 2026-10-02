# 46개의 토픽으로 분류된 11,228개의 로이터 뉴스 기사 텍스트 데이터셋 다중분류

from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, # 단어사전의 개수 # 빈도수가 높은 단어 순으로 1000개를 뽑음 (default None)
    # maxlen=100, # 문서의 최대 길이를 제한하며, 초과 시 자름
    test_split=0.2,
)

print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape) # (2246,) (2246,)

print(x_train)
print(min(min(x) for x in x_train)) # 1
print(max(max(x) for x in x_train)) # 999

print(y_train) # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train)) # [ 0  1  2 ... 44 45]

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'numpy.ndarray'>

############ 뉴스 기사 길이 분석 ############
print('뉴스 기사의 최대 길이 :', max(len(sample) for sample in x_train)) # 뉴스 기사의 최대 길이 : 2376
print('뉴스 기사의 최소 길이 :', min(len(sample) for sample in x_train)) # 뉴스 기사의 최소 길이 : 13
print('뉴스 기사의 평균 길이 :', sum(map(len, x_train))/len(x_train)) # 뉴스 기사의 평균 길이 : 145.5398574927633

plt.hist([len(sample) for sample in x_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# padding
x_train = pad_sequences(x_train, 
                         padding="pre", # 앞쪽을 0으로 채움 # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                         maxlen=250, # 모든 문장을 토큰 250개 길이로 맞춤 (지정하지 않으면 가장 긴 시퀀스 기준)
                         truncating='post' # 문장이 길면 뒤쪽 토큰부터 제거
                         )
print(x_train)
print(x_train.shape) # (8982, 250)

x_test = pad_sequences(x_test, 
                         padding="pre", # 앞쪽을 0으로 채움 # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                         maxlen=250, # 모든 문장을 토큰 250개 길이로 맞춤 (지정하지 않으면 가장 긴 시퀀스 기준)
                         truncating='post' # 문장이 길면 뒤쪽 토큰부터 제거
                         )
print(x_test)
print(x_test.shape) # (2246, 250)

# y원핫(안함)


# 2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))
model.add(LSTM(100))
model.add(Dense(46, activation="softmax"))


# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="adam", 
    metrics=["acc"]
)
start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=1000,
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
print("acc :", round(loss[1], 4)) # acc 0.67 목표
print("===========================")

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택

# print(classification_report(y_test, y_pred))
print("걸린시간 :", round(end_time - start_time), "초")

# ===========================
# loss : 1.1078989505767822
# acc : 0.7422
# ===========================
# 71/71 [==============================] - 1s 6ms/step
# 걸린시간 : 49 초