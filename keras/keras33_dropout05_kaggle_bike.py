# 31-5 카피

# 캐글 Bike Sharing Demand
# https://www.kaggle.com/competitions/bike-sharing-demand/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
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
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True, # 성능이 가장 좋았던 Epoch의 가중치를 모델에 다시 적용 # default는 False
)

date = datetime.datetime.now().strftime("%m%d_%H%M_")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join(["./_save/keras33/", "k33_05_", date, filename]) # 모델 저장 경로

mcp = ModelCheckpoint( # 모델 구조 + val_loss가 가장 낮은 학습된 가중치 저장
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath, # 저장할 경로
    verbose=1,
)

model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=24, 
                 validation_data=(x_valid, y_valid),
                 callbacks=[es, mcp],
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

# ####################### 제출용 #################33
y_submit = model.predict(test_csv)
submission['count'] = y_submit
# print(submission)
# print(submission.shape)

# submission.loc[submission['count'] < 0, 'count'] = 0

filename = datetime.datetime.now().strftime("submit_%m%d_%H%M.csv")
submission.to_csv(path + "submit/" + filename)


import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][5:], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("캐글 자전거 수요 Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()

'''
# 1차 시도 submit_0904_1615
random_stat = 42
test_size=0.2
epochs = 500000
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
RMSLE : 1.29108
'''

'''
# 4차 시도 submit_0910_1709

MinMaxScaler 적용

random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
모든 activation='relu'

EarlyStopping
monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
r2 : 0.3092743754386902
mse : 22798.736328125
rmse : 150.99250421171575
걸린시간 :  125.07 초
# 캐글 점수
RMSLE : 1.34263
'''

'''
# 5차 시도 submit_0911_1331

StandardScaler 적용

random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
모든 activation='relu'

EarlyStopping
monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
r2 : 0.3498837351799011
mse : 21458.345703125
rmse : 146.4866741486235
걸린시간 :  170.89 초
# 캐글 점수
RMSLE : 1.33843
'''

'''
# 6차 시도 submit_0911_1533

MaxAbsScaler 적용

random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
모든 activation='relu'

EarlyStopping
monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
r2 : 0.2973331809043884
mse : 23192.876953125
rmse : 152.2920777753229
걸린시간 :  139.84 초
# 캐글 점수
RMSLE : 1.35244
'''

'''
#####################################
# 6차 시도 submit_0911_1649

RobustScaler 적용

random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
모든 activation='relu'

EarlyStopping
monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
r2 : 0.3461220860481262
mse : 21582.505859375
rmse : 146.90985623631587
걸린시간 :  304.19 초
# 캐글 점수
RMSLE : 
####################################
'''

'''
#####################################
# 7차 시도 submit_0914_1624

RobustScaler 적용 (ModelCheckpoint)

random_stat = 42
test_size=0.2
epochs = 500
batch_size = 24
모든 activation='relu'

EarlyStopping
monitor='val_loss',
mode='min',
patience=50,
restore_best_weights=True,

# 결과
r2 : 0.3585832715034485
mse : 21171.201171875
rmse : 145.50326859515906
걸린시간 :  273.16 초
# 캐글 점수
RMSLE : 
####################################
'''