# keras48에서 저장한 my pet.jpg의 npy를 불러와서
# keras44에서 저장한 cat dog 모델로 예측해보기 

import numpy as np
from tensorflow.keras.models import load_model

# 데이터 불러오기
np_path = "./_data/image/"
x = np.load(np_path + "keras48_mypet.npy")
x = x/255. # 스케일링

# 모델 불러오기
model_path ="./_save/keras44/"
model = load_model(model_path + "k44_03_0918_1825_0037-0.2136.keras") # cat dog 모델

# 예측하기
y_pred = model.predict(x)
print(y_pred) # [[0.69043756]]
y_pred = np.round(y_pred)
print(y_pred) # [[1.]]

# class_indices : {'cats': 0, 'dogs': 1}
if y_pred == 1:
    print("개 이미지~")
else:
    print("고양이 이미지~")