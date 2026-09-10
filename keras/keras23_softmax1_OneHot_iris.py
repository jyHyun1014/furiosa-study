import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESC)
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (150, 4) (150,)
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))



# ########### 원핫 1. to_categorical ############
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# # [[1. 0. 0.]
# #  [1. 0. 0.]
# #  [1. 0. 0.]
# #  ...
# print(y.shape) # (150, 3)

# ########### 원핫 2. pd.get_dummies ############
# y = pd.get_dummies(y, columns = ['target'], dtype=int, drop_first=True)
# print(y)
# #          1      2
# # 0    False  False
# # 1    False  False
# # ..     ...    ...
# # 148  False   True
# # 149  False   True
# print(y.shape) # (150, 2)

########### 원핫 3. OneHotEncoder ############
from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder() # sparse(희소행렬) 형태로 반환함 # 0을 굳이 전부 저장하지 않고, 1이 있는 위치만 저장
ohe = OneHotEncoder(sparse_output=False) # 배열 그대로 반환
y = ohe.fit_transform(y.reshape(-1,1)) # y의 shape가 (150,)인 벡터 상태이지만 (150, 1)로 변환 후 넣어야 함
print(y.toarray())
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  ...
print(y.shape) # (150, 3)

exit()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, y_train.shape) # (120, 4) (120, 3)
print(x_test.shape, y_test.shape) # (30, 4) (30, 3)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', # 정수 레이블인 경우 sparse_categorical_crossentropy
    optimizer='adam', 
    metrics=['acc'],
    )
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", round(result[1], 2))

y_pred = model.predict(x_test)
print(y_pred[:3])
# [[2.4020884e-04 4.8378080e-01 5.1597905e-01]
#  [9.9980766e-01 1.8809446e-04 4.3173541e-06]
#  [5.4083830e-03 9.9352044e-01 1.0711314e-03]
#  ...
y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
y_test = np.argmax(y_test, axis=1) # 가장 확률이 높은 인덱스 선택
print(y_pred) # [2 0 1 1 2 1 2 2 0 1 0 0 0 1 2 2 2 1 1 0 2 0 1 0 0 1 1 2 0 2]

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("걸린시간 :", round(end_time - start_time), "초")