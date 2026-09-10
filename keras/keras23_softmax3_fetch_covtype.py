import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# acc 0.93

# 1. 데이터
datasets = fetch_covtype()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
# ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

print(y[:3]) # [5 5 2]
########### 원핫 1. to_categorical ############
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y[:3])
# [[0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  ...
print(y.shape) # (581012, 8)


# y값이 1부터 시작한다면, 원핫인코딩을 하면 0부터 생김
# 그래야지 argmax 하면 정상적으로 디코딩 가능
# 그런데 이 방법은 학습할때 연산 낭비가 심해서 그냥 첫번째 열 drop하고 디코딩할때 +1 하는게 더 효율적임


# print(y) # [5 5 2 ... 3 3 3]
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(y) # [4 4 1 ... 2 2 2]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99, stratify=y)
# print(x_train.shape, y_train.shape) # (464809, 54) (464809,)
# print(x_test.shape, y_test.shape) # (116203, 54) (116203,)


# # 2. 모델 구성
# model = Sequential()
# model.add(Dense(20, activation='swish', input_dim=54))
# model.add(Dense(20, activation='swish'))
# model.add(Dense(15, activation='swish'))
# model.add(Dense(10, activation='swish'))
# model.add(Dense(7, activation='softmax'))

# # 3. 컴파일, 훈련
# model.compile(
#     loss='sparse_categorical_crossentropy', # 정수 레이블인 경우 sparse_categorical_crossentropy
#     optimizer='adam', 
#     metrics=['acc'],
#     )
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=30,
#     restore_best_weights=True,
# )
# start_time = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=1000,
#           verbose=1,
#           validation_split=0.2,
#           callbacks=[es],
#           )
# end_time = time.time()


# # 4. 평가, 예측
# result = model.evaluate(x_test, y_test)
# print("loss :", result[0])
# print("acc :", round(result[1], 2))

# y_pred = model.predict(x_test)
# # print(y_pred)
# # [[4.2225206e-01 5.4432935e-01 8.1924336e-05 ... 2.0792518e-02
# #   7.6491653e-04 1.1779276e-02]
# #  ...

# y_pred = np.argmax(y_pred, axis=1) # 가장 확률이 높은 인덱스 선택
# y_pred = le.inverse_transform(y_pred)
# y_test = le.inverse_transform(y_test)
# print(y_pred) # [1 2 2 ... 1 2 2]

# from sklearn.metrics import confusion_matrix, classification_report

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print("걸린시간 :", round(end_time - start_time), "초")