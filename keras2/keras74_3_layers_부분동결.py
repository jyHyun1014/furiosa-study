# 74-2 카피

# ImageNet으로 사전학습된 VGG16과 추가한 Dense 층 중 일부의 가중치를 동결하고, 
# 전체 가중치와 학습 가능한 가중치의 개수 및 각 Layer의 학습 가능 여부를 확인하기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False

model.summary()

print(len(model.weights)) # 36 # (vgg16에서 FC제외한 layer 13개 + 내가 추가한 Dense layer 5개) * weight, bias 각각 2개
print(len(model.trainable_weights)) # 8

import pandas as pd
pd.set_option("max_colwidth", None)
layers = [(type(layer).__name__, layer.name, layer.trainable) for layer in model.layers]
print(layers) # [('Functional', 'vgg16', False), ('Flatten', 'flatten', True), ('Dense', 'dense', False), ('Dense', 'dense_1', True), ('Dense', 'dense_2', True), ('Dense', 'dense_3', True), ('Dense', 'dense_4', True)]

results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense            False
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True