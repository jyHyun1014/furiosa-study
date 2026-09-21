# 데이터 넘파이 저장
# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import datetime

path_data = './_data/image/man_woman/'

# 1. 데이터
datagen = ImageDataGenerator(
    rescale=1/255.,
    # rotation_range=5, # 회전
    # width_shift_range=0.1, # 가로 이동
    # height_shift_range=0.1, # 세로 이동
    # zoom_range=0.2, # 1 ± 0.2 확대/축소
    # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    # fill_mode='nearest', # 변환 후 빈 영역 처리
    # validation_split=0.2,
)

xy = datagen.flow_from_directory( # Found 27167 images belonging to 2 classes.
    path_data, # 경로
    target_size=(100,100),
    batch_size=10000000,
    class_mode='binary', # 이진분류
    color_mode='rgb', # 컬러
    shuffle=False,
)
print('class_indices :', xy.class_indices) # class_indices : {'man': 0, 'woman': 1}

x = xy[0][0]
y = xy[0][1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (21733, 100, 100, 3) (21733,) (5434, 100, 100, 3) (5434,)

# np array 저장
np_path = "./_save/keras46/"
np.save(np_path + "keras46_03_x_train.npy", arr=x_train)
np.save(np_path + "keras46_03_y_train.npy", arr=y_train)
np.save(np_path + "keras46_03_x_test.npy", arr=x_test)
np.save(np_path + "keras46_03_y_test.npy", arr=y_test)