# 50-1 카피

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],

# # 사진 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'grey')
# plt.show()

########## 여기부터 증폭 ##########
datagen = ImageDataGenerator(
    rescale=1/255., # 스케일링
    rotation_range=5, # 회전
    width_shift_range=0.1, # 가로 이동
    height_shift_range=0.1, # 세로 이동
    zoom_range=0.1, # 1 ± 0.1 확대/축소
    # shear_range=0.7, # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # horizontal_flip = True, # 좌우 반전
    # vertical_flip=True, # 상하 반전
    fill_mode='nearest', # 변환 후 빈 영역 처리
)

augment_size = 100

print(x_train.shape) # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)


#### np.tile 테스트 ####
a = np.array([[1, 2],[3, 4]])
print(a)
# [[1 2]
#  [3 4]]
print(a.shape) # (2, 2)
a = np.tile(a, 2) # 입력한 배열 전체를 지정한 횟수만큼 반복하여 새로운 배열을 만드는 함수
print(a)
# [[1 2 1 2]
#  [3 4 3 4]]
print(a.shape) # (2, 4)
a = a.reshape(-1, 2, 2, 1)
print(a)
# [[[[1]
#    [2]]

#   [[1]
#    [2]]]


#  [[[3]
#    [4]]

#   [[3]
#    [4]]]]
print(a.shape) # (2, 2, 2, 1)
#### np.tile 테스트 ####


xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1), # x # .reshape(28*28)을 안넣으면 가로로 tile 복붙이 되기 때문에 꼭 넣어야 함
    np.zeros(augment_size), # y
    batch_size=augment_size,
    shuffle=False,
).next()

print(xy_data)
print(type(xy_data)) # <class 'tuple'>
print(len(xy_data)) # 2 # x와 y

print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()