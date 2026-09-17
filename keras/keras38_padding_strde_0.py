import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# 2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), # 출력크키 (10, 10, 10)
                 strides=1, # 디폴트
                 padding='same',
                 ))

model.add(Conv2D(9, kernel_size=(3,3), # 출력크키 (4, 4, 9)
                 strides=2,
                 padding='valid', # 디폴트
                 ))

model.summary()