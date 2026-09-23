# 35-5 카피

# 캐글 Bike Sharing Demand
# https://www.kaggle.com/competitions/bike-sharing-demand/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import time
import datetime

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

# 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
test_csv = scaler.transform(test_csv)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu')) # activation의 default는 linear
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=100, 
                 batch_size=24, 
                 validation_data=(x_valid, y_valid),
                 )
end_time = time.time() # 현재시간을 반환. 끝시간

loss = model.evaluate(x_valid, y_valid)
y_valid_predict = model.predict(x_valid)
r2 = r2_score(y_valid, y_valid_predict)
mse = mean_squared_error(y_valid, y_valid_predict)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_predict))
print("r2 :", r2)
print("mse :", mse)
print("rmse :", rmse)
print("걸린시간 : ", round(end_time - start_time, 2), "초") # 2번째 자리에서 반올림

# CPU
# r2 : 0.2814326882362366
# mse : 23717.705078125
# rmse : 154.00553586843884
# 걸린시간 :  35.02 초

# GPU
# r2 : 0.27730321884155273
# mse : 23854.005859375
# rmse : 154.44742101885353
# 걸린시간 :  64.54 초

# learning_rate=0.01
# r2 : 0.19650042057037354
# mse : 26521.05859375
# rmse : 162.85287407273475
# 걸린시간 :  65.52 초