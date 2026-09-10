from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃ (입력 개수 * weight 개수) + bias 개수
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 3)                   │               6 │ (1 * 3) + 3 = 6
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 4)                   │              16 │ (3 * 4) + 4 = 16
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 3)                   │              15 │ (4 * 3) + 3 = 15
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 1)                   │               4 │ (3 * 1) + 1 = 4
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 41 (164.00 B)
#  Trainable params: 41 (164.00 B)
#  Non-trainable params: 0 (0.00 B)