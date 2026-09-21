# 모델 불러와서 내 사진으로 예측

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


path = "c:/study/_data/image/"
img = load_img(path + "my pet.jpg", target_size=(100,100))

print(img) # <PIL.Image.Image image mode=RGB size=150x150 at 0x29B6139F7F0>
print(type(img)) # <class 'PIL.Image.Image'>

# 이미지 확인하기
plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) # (100, 100, 3)
print(type(arr)) # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 차원 증가
print(arr)
print(arr.shape) # (1, 100, 100, 3)

arr = arr/255. # 스케일링


# np_path = "./_data/image/"
# np.save(np_path + "keras49_mypet.npy", arr=arr)

# 모델 불러오기
model_path ="./_save/keras47/"
model = load_model(model_path + "k47_03_0921_1519_0087-0.3005.keras") # man woman 모델

# 예측하기
y_pred = model.predict(arr)
print(y_pred) # [[0.5118274]]
y_pred = np.round(y_pred)
print(y_pred) # [[1.]]

# class_indices : {'man': 0, 'woman': 1}
if y_pred == 1:
    print("woman 이미지~")
else:
    print("man 이미지~")