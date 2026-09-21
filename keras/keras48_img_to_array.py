from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
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

np_path = "./_data/image/"
np.save(np_path + "keras48_mypet.npy", arr=arr)