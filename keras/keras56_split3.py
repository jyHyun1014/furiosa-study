import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
print(x_predict) # [ 96  97  98  99 100 101 102 103 104 105]

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10-3+1 = range(0, 8)
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(a, size)
# print(bbb)
# [[  1   2   3   4   5   6]
#  [  2   3   4   5   6   7]
# ...
#  [ 94  95  96  97  98  99]
#  [ 95  96  97  98  99 100]]
print(bbb.shape) # (95, 6)

x_predict = split_x(x_predict, size-1)
print(x_predict)
# [[ 96  97  98  99 100]
#  [ 97  98  99 100 101]
#  [ 98  99 100 101 102]
#  [ 99 100 101 102 103]
#  [100 101 102 103 104]
#  [101 102 103 104 105]]

# x와 y 분리하기
x = bbb[: , :-1]
y = bbb[: , -1]
print(x)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
# ...
#  [94 95 96 97 98]
#  [95 96 97 98 99]]
print(x.shape) # (95, 5)
print(y)
# [  6   7 ... 99 100]
print(y.shape) # (95,)


# 모델에 넣기 위해 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (95, 5, 1) # (batch_size, timesteps, features)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print(x_predict.shape) # (6, 5, 1) # (batch_size, timesteps, features)


# 2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(5, 1))) # RNN 계열의 끝판왕
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
# loss는 0.1 이하
# 결과는 [101, 102, 103, 104, 105, 106] 의 근사치가 나오면 됨

# loss : 0.01850828528404236
# 1/1 [==============================] - 0s 211ms/step
# x_predict의 결과 : [[100.80104 ]
#  [101.57446 ]
#  [102.312836]
#  [103.00139 ]
#  [103.64476 ]
#  [104.260895]]