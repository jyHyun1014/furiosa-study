# 캐글 Bike Sharing Demand
# https://www.kaggle.com/competitions/bike-sharing-demand/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

# 1. 데이터
path = "c:/study/_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
print(submission) # [6493 rows x 1 columns]

##### 결측치 확인 #####
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe().T)

print(train_csv.isna().sum())
print(test_csv.isna().sum())

##### x, y 분리 #####
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y) # (10886,)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu')) # activation의 default는 linear
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=24)

loss = model.evaluate(x_valid, y_valid)
y_valid_predict = model.predict(x_valid)
r2 = r2_score(y_valid, y_valid_predict)
mse = mean_squared_error(y_valid, y_valid_predict)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_predict))
print("r2 :", r2)
print("mse :", mse)
print("rmse :", rmse)
# r2 : 0.27376800775527954
# mse : 23970.69140625
# rmse : 154.82471187200704

####################### 제출용 #################33
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=24)


y_submit = model.predict(test_csv)
submission['count'] = y_submit
# print(submission)
# print(submission.shape)

# submission.loc[submission['count'] < 0, 'count'] = 0

submission.to_csv(path + "submit/" + "submit_0904_1652.csv")

'''
# 1차 시도 submit_0904_1615
random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
# 결과
r2 : 0.27376800775527954
mse : 23970.69140625
rmse : 154.82471187200704
# 캐글 점수
RMSLE : 4.76188
'''

'''
# 3차 시도 submit_0904_1652
random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
출력층을 제외한 모든 activation='relu'
# 결과
r2 : 0.31050413846969604
mse : 22758.14453125
rmse : 150.85802773220257
# 캐글 점수
RMSLE : 
'''

