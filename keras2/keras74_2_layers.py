# 74-1 카피

# ImageNet으로 사전학습된 VGG16의 가중치를 동결하고 Flatten과 Dense 층을 추가한 뒤, 
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
model.add(Dense(10, activation='softmax'))

print(len(model.weights)) # 30 # (vgg16에서 FC제외한 layer 13개 + 내가 추가한 Dense layer 2개) * weight, bias 각각 2개
print(len(model.trainable_weights)) # 4

import pandas as pd
pd.set_option("max_colwidth", None)
layers = [(type(layer).__name__, layer.name, layer.trainable) for layer in model.layers]
print(layers) # [('Functional', 'vgg16', False), ('Flatten', 'flatten', True), ('Dense', 'dense', True), ('Dense', 'dense_1', True)]

results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True