import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42,
    )
print("x_train :", x_train) 
print("x_test :", x_test)
print("y_train :", y_train)
print("y_test :", y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

print("=======================================")
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)   # loss : 21.09121322631836
results = model.predict(x)
print(results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)   # 데이터 점찍기
plt.plot(x, results, color='red')
plt.show()