import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# acc : 1.0

# 1. 데이터
datasets = load_digits()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names) # ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64) (1797,)
print(y) # [0 1 2 ... 8 9 8]
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
print(x_test.shape, y_test.shape) # (360, 64) (360,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(20, activation='swish', input_dim=64))
model.add(Dense(20, activation='swish'))
model.add(Dense(15, activation='swish'))
model.add(Dense(15, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'],
)
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=100000,
    batch_size=10,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", round(result[1], 2))

y_pred = model.predict(x_test)
print(y_pred)
# [[3.32570425e-03 5.96013200e-03 5.46997107e-05 ... 1.79968663e-02
#   8.52080266e-05 1.09515987e-01]
#  ...

y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
print(y_pred) # [5 2 8 1 7 2 6 2 6 5 0 5 9 ...

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("걸린시간 :", round(end_time - start_time), "초")