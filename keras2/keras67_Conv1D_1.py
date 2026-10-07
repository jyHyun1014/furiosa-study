# 54-1 카피

# 시계열 데이터를 Conv1D로 모델링

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (7, 3, 1)

# 2. 모델 구성
model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1))) # (None, 2, 10)
model.add(Conv1D(10, 2)) # (None, 1, 10)

model.add(Flatten())
# model.add(GlobalAveragePooling1D())

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# 3. 컴파일
es = EarlyStopping(
    monitor="val_loss",
    patience=30,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau( # val_loss가 10 epoch 동안 개선되지 않으면 Learning Rate를 절반으로 줄임
    monitor="val_loss",
    mode="auto",
    patience=20,
    verbose=1,
    factor=0.5,
)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
model.fit(x, y, 
          epochs=1000000,
          batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, reduce_lr],
          )

# 4. 평가, 예측
results= model.evaluate(x, y)
print("loss :", results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print("[8,9,10]의 결과 :", y_predict)
# [8,9,10]의 결과 : [[11.732902]]