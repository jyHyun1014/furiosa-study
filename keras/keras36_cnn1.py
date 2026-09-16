from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(10, 10, 1)))
model.add(Conv2D(5, kernel_size=(2,2)))

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 8, 8, 10)            │             100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 7, 7, 5)             │             205 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 305 (1.19 KB)
#  Trainable params: 305 (1.19 KB)
#  Non-trainable params: 0 (0.00 B)