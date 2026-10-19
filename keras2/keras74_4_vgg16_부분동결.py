# 74-3 카피

# ImageNet으로 사전학습된 VGG16 모델의 특정 층만 선택적으로 동결하고, 
# 각 Layer의 학습 가능 여부를 확인하기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)

model.layers[11].trainable = False

model.summary()

import pandas as pd
pd.set_option("max_colwidth", None)
layers = [(type(layer).__name__, layer.name, layer.trainable) for layer in model.layers]
print(layers)
# [('InputLayer', 'input_1', True), ('Conv2D', 'block1_conv1', True), ('Conv2D', 'block1_conv2', True), ('MaxPooling2D', 'block1_pool', True), ('Conv2D', 'block2_conv1', True), ('Conv2D', 'block2_conv2', True), ('MaxPooling2D', 'block2_pool', True), ('Conv2D', 'block3_conv1', True), ('Conv2D', 'block3_conv2', True), ('Conv2D', 'block3_conv3', True), ('MaxPooling2D', 'block3_pool', True), ('Conv2D', 'block4_conv1', False), ('Conv2D', 'block4_conv2', True), ('Conv2D', 'block4_conv3', True), ('MaxPooling2D', 'block4_pool', True), ('Conv2D', 'block5_conv1', True), ('Conv2D', 'block5_conv2', True), ('Conv2D', 'block5_conv3', True), ('MaxPooling2D', 'block5_pool', True)]

results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#       Layer Type    Layer Name  Layer Trainable
# 0     InputLayer       input_1             True
# 1         Conv2D  block1_conv1             True
# 2         Conv2D  block1_conv2             True
# 3   MaxPooling2D   block1_pool             True
# 4         Conv2D  block2_conv1             True
# 5         Conv2D  block2_conv2             True
# 6   MaxPooling2D   block2_pool             True
# 7         Conv2D  block3_conv1             True
# 8         Conv2D  block3_conv2             True
# 9         Conv2D  block3_conv3             True
# 10  MaxPooling2D   block3_pool             True
# 11        Conv2D  block4_conv1            False
# 12        Conv2D  block4_conv2             True
# 13        Conv2D  block4_conv3             True
# 14  MaxPooling2D   block4_pool             True
# 15        Conv2D  block5_conv1             True
# 16        Conv2D  block5_conv2             True
# 17        Conv2D  block5_conv3             True
# 18  MaxPooling2D   block5_pool             True