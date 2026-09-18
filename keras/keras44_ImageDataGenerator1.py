import numpy as np
from keras.preprocessing.image import ImageDataGenerator

print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale=1/255., # 스케일링
    rotation_range=5, # 회전
    width_shift_range=0.1, # 가로 이동
    height_shift_range=0.1, # 세로 이동
    zoom_range=0.2, # 1 ± 0.2 확대/축소
    shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    horizontal_flip = True, # 좌우 반전
    vertical_flip=True, # 상하 반전
    fill_mode='nearest', # 변환 후 빈 영역 처리
)
test_datagen = ImageDataGenerator(
    rescale=1/255.,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory( # Found 160 images belonging to 2 classes.
    path_train, # 경로
    target_size=(100,100),
    batch_size=10, # 16 * (10, 100, 100, 1)
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory( # Found 120 images belonging to 2 classes
    path_test, # 경로
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False, # test에서는 필요가 없다
)

print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x0000020BF0EA7FA0>
# print(xy_train.next()) # 이터레이터의 첫번째를 보여줘!
# print(xy_train.next()) # 두번째 이터레이터를 출력해줘

# print(xy_train[0]) # 첫번째 배치의 xy
# print(xy_train[1]) # 두번째 배치의 xy
# print(xy_train[2]) # 세번째 배치의 xy

# print(xy_train[0][0]) # 첫번째 배치의 x
# print(xy_train[0][1]) # 첫번째 배치의 y
# print(xy_train[16][0]) # 여기부터 에러. 이유는 160장, 배치는 10이니까

print(xy_train[0][0].shape) # (10, 100, 100, 1)
print(xy_train[0][1].shape) # (10,)

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>