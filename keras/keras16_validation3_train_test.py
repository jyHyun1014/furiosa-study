import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 8개, 4개, 4개
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=False)

print(x_train, y_train)
print(x_val, y_val)
print(x_test, y_test)