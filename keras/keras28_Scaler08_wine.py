import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import time

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

########### 원핫 1. to_categorical ############
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  ...
print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99, stratify=y)
print(x_train.shape, y_train.shape) # (142, 13) (142, 3)
print(x_test.shape, y_test.shape) # (36, 13) (36, 3)

# 스케일링
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='swish', input_dim=13))
model.add(Dense(10, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(5, activation='swish'))
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
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
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
y_test = np.argmax(y_test, axis=1) # 가장 확률이 높은 인덱스 선택
print(y_pred) # [1 1 2 1 1 1 2 1 0 0 0 0 2 0 2 0 1 2 1 1 1 1 2 2 1 1 2 0 1 0 2 1 1 0 0 0]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("걸린시간 :", round(end_time - start_time), "초")
# [[12  0  0]
#  [ 0 14  0]
#  [ 0  0 10]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        12
#            1       1.00      1.00      1.00        14
#            2       1.00      1.00      1.00        10

#     accuracy                           1.00        36
#    macro avg       1.00      1.00      1.00        36
# weighted avg       1.00      1.00      1.00        36

# 걸린시간 : 13 초

"""
##########################################
MaxAbsScaler 적용
loss : 0.22317510843276978
acc : 0.97
걸린시간 : 62 초
###########################################

"""