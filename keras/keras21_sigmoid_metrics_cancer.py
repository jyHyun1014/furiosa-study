import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) # 데이터 설명

x = datasets.data
# x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x)) # <class 'numpy.ndarray'>

# Pandas : NumPy를 기반(Wrapper)으로 만들어진 라이브러리
# Pandas의 핵심 데이터 구조인 DataFrame과 Series는 내부적으로 NumPy의 다차원 배열(ndarray)을 사용하여 데이터를 저장하고 관리함

print(y)
print(type(y)) # <class 'numpy.ndarray'>
print(np.unique(y)) # [0 1]
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([170, 285]))
print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([42, 72]))
print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape) # (114, 30) (114,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류는 출력층 무조건 sigmoid

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', # 가중치 업데이트에 사용
              optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['acc'], # 성능 확인용
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)
start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=5, 
    validation_split=0.2, 
    callbacks=[es]
    )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # loss : [0.18545010685920715, 0.9298245906829834] # compile()의 [loss, metrics] 순서대로 반환
print("===========================") # loss : 0.18545010685920715
print("loss :", loss[0]) #
print("acc :", round(loss[1], 4)) # acc : 0.9298
print("===========================")

y_pred = model.predict(x_test) # 0과 1사이의 실수 값으로 나옴
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred[:5])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)


print("걸린시간 : ", round(end_time - start_time, 2), "초")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y 값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title("Cancer Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() # 격자표시 추가
plt.show()