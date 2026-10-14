# Dense 모델의 가중치와 학습 가능 파라미터를 확인하고, 모델 동결(Freeze) 전후의 변화를 비교하기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print(tf.__version__)

# 난수 시드 고정
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5,])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6                                                                 
#  dense_1 (Dense)             (None, 2)                 8                                                                   
#  dense_2 (Dense)             (None, 1)                 3                                                                  
# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________

print("========================================================= model.weights =========================================================")
print(model.weights) # 모델의 가중치 확인
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, 
#  <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[-0.92561173,  0.8256177 ],[ 0.6200088 ,  1.0182774 ],[-0.5191052 , -0.6304303 ]], dtype=float32)>, 
#  <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[-0.02628279],[-1.074922  ]], dtype=float32)>, 
#  <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# model.weights
# - 모델이 보유한 모든 가중치 변수(kernel, bias)를 반환
# - 학습 가능 여부와 관계없이 모든 가중치 변수를 포함
# - 각 Dense 레이어마다 kernel(가중치)과 bias(편향)가 존재


print("========================================================= model.trainable weights =========================================================")
print(model.trainable_weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, 
#  <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[-0.92561173,  0.8256177 ],[ 0.6200088 ,  1.0182774 ],[-0.5191052 , -0.6304303 ]], dtype=float32)>, 
#  <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[-0.02628279],[-1.074922  ]], dtype=float32)>, 
#  <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# model.trainable_weights
# - 모델에서 학습 가능한 가중치 변수만 반환
# - 학습 과정에서 optimizer가 업데이트할 수 있는 가중치
# - 현재 모든 레이어가 trainable=True(기본값)이므로 model.weights와 동일

print("=========================================================")
print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 6

# len(model.weights)는 전체 파라미터 개수(17)가 아니라
# 가중치 변수(kernel, bias)의 개수(6)를 반환함

print("========================================================= 동결 =========================================================")
model.trainable = False # 모델 동결(Freeze)
print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 0

# model.weights           : 모든 가중치가 그대로 존재하므로 6개
# model.trainable_weights : 학습 가능한 가중치가 없으므로 0개

print("========================================================= model.weights =========================================================")
print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, 
#  <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[-0.92561173,  0.8256177 ],[ 0.6200088 ,  1.0182774 ],[-0.5191052 , -0.6304303 ]], dtype=float32)>, 
#  <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
#  <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[-0.02628279],[-1.074922  ]], dtype=float32)>, 
#  <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# 동결 전과 동일하게 모든 가중치가 존재함
# 동결은 가중치를 제거하는 것이 아니라 학습 대상에서 제외하는 것

print("========================================================= model.trainable weights =========================================================")
print(model.trainable_weights) # []
# 모든 레이어가 동결되어 학습 가능한 가중치가 없음

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6                                                               
#  dense_1 (Dense)             (None, 2)                 8                                                              
#  dense_2 (Dense)             (None, 1)                 3                                                              
# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# _________________________________________________________________

# 모델 구조와 전체 파라미터 개수는 동일하지만
# 모든 파라미터가 Non-trainable로 변경됨