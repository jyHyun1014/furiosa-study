from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(4, kernel_size=(2,2), input_shape=(5, 5, 1))) # 
model.add(Conv2D(3, kernel_size=(2,2)))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #    (((커널사이즈) * 채널수) + bias) * 필터개수
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 4)           20        ((2 * 2) + 1) * 4
                                                                 
#  conv2d_1 (Conv2D)           (None, 3, 3, 3)           51        ((2 * 2) * 4) + 1) * 3
                                                                 
# =================================================================
# Total params: 71
# Trainable params: 71
# Non-trainable params: 0
# _________________________________________________________________