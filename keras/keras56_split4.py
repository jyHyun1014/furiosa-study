# 데이터를 reshape 한 후, split_x 함수로 시계열 데이터로 변환 (N, 5, 2)

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
print(x_predict) # [ 96  97  98  99 100 101 102 103 104 105]

a = a.reshape(-1, 2)
print(a)
# [[  1   2]
#  [  3   4]
# ...
#  [ 97  98]
#  [ 99 100]]
print(a.shape) # (50, 2)

x_predict = x_predict.reshape(-1, 2)
print(x_predict)
# [[ 96  97]
#  [ 98  99]
#  [100 101]
#  [102 103]
#  [104 105]]
print(x_predict.shape) # (5, 2)

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10-3+1 = range(0, 8)
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[[  1   2]
#   [  3   4]
#   [  5   6]
#   [  7   8]
#   [  9  10]
#   [ 11  12]]
# ...
#  [[ 89  90]
#   [ 91  92]
#   [ 93  94]
#   [ 95  96]
#   [ 97  98]
#   [ 99 100]]]
print(bbb.shape) # (45, 6, 2)

x_predict = split_x(x_predict, size-1)
print(x_predict)
# [[[ 96  97]
#   [ 98  99]
#   [100 101]
#   [102 103]
#   [104 105]]]
print(x_predict.shape) # (1, 5, 2)


# x와 y 분리하기
x = bbb[: , :-1]
y = bbb[: , -1, -1]
print(x)
# [[[  1   2]
#   [  3   4]
#   [  5   6]
#   [  7   8]
#   [  9  10]
#   [ 11  12]]
# ...
#  [[89 90]
#   [91 92]
#   [93 94]
#   [95 96]
#   [97 98]]]
print(x.shape) # (45, 5, 2)
print(y) # [12  14 ... 98 100]
print(y.shape) # (45,)


# 2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(5, 2)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일
es = EarlyStopping(
    monitor="loss",
    patience=30,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, 
          epochs=1000000,
          batch_size=1,
          verbose=1,
          callbacks=[es],
          )

# 4. 평가, 예측
results= model.evaluate(x, y)
print("loss :", results)

y_predict = model.predict(x_predict)
print("x_predict의 결과 :", y_predict)

# loss : 0.034566305577754974
# 1/1 [==============================] - 0s 196ms/step
# x_predict의 결과 : [[104.07438]]