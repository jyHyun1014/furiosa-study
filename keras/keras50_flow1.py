from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

path = "c:/study/_data/image/"
img = load_img(path + "my pet.jpg", target_size=(150,150))

print(img) # <PIL.Image.Image image mode=RGB size=150x150 at 0x29B6139F7F0>
print(type(img)) # <class 'PIL.Image.Image'>

# 이미지 확인하기
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) # (150, 150, 3)
print(type(arr)) # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 차원 증가
print(arr)
print(arr.shape) # (1, 150, 150, 3)

# np_path = "./_data/image/"
# np.save(np_path + "keras48_mypet.npy", arr=arr)

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

it = datagen.flow(
    arr,
    batch_size=1,   
)
print(it) # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000023375547FA0>

# next() 호출할 때마다 원본 이미지에 랜덤한 변환을 적용해 새로운 batch를 생성하여 반환함
# print(it.next()) # python 3.10까지
print(next(it)) # python 3.11이후
print(next(it).shape) # (1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    batch = batch.reshape(150, 150, 3)

    ax[i].imshow(batch)
    # ax[i].axis('off')
plt.show()