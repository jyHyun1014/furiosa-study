# 문장을 단어 단위로 분리하고, 각 단어에 정수 번호를 부여하는 클래스
from tensorflow.keras.preprocessing.text import Tokenizer 

# 분석할 문장
text = '나는 지금 매우 매우 맛있는 김밥을 엄청 많이 많이 먹었다'

# Tokenizer 객체 생성
token = Tokenizer() # 객체(인스턴스) = 클래스() = 인스턴스 생성

# 문장을 학습시켜 단어 빈도를 세고, 단어별 정수 번호를 부여
token.fit_on_texts([text]) # 대괄호로 감싸는 이유: fit_on_texts()는 문장들의 리스트를 입력으로 받기 때문

# 단어별 정수 번호 출력
# 빈도가 높은 단어부터 작은 번호가 부여됨
# 빈도가 같으면 문장에 먼저 등장한 단어가 앞 번호를 받음
print(token.word_index)
# {'매우': 1, '많이': 2, '나는': 3, '지금': 4, '맛있는': 5, '김밥을': 6, '엄청': 7, '먹었다': 8}

# 단어별 등장 횟수 출력
print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('많이', 2), ('먹었다', 1)])

# 원래 문장을 Tokenizer가 부여한 단어 번호 시퀀스로 변환
# 바깥 리스트는 문장 단위, 안쪽 리스트는 각 문장의 토큰 번호
x = token.texts_to_sequences([text])
print(x) # [[3, 4, 1, 1, 5, 6, 7, 2, 2, 8]]
# 매우(1)를 2배하면 많이(2)가 되면 안되기 때문에 문제 발생
print(len(x[0])) # 10
# 문장 1개에 토큰 10개가 있다는 뜻

########### 원핫 1. to_categorical ############
from tensorflow.keras.utils import to_categorical

y = to_categorical(x)
print(y)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
# ...
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
print(y.shape) # (1, 10, 9) # (문장 수, 토큰 수, 클래스 수)
# 무조건 0부터 시작함
# 최대 토큰 번호가 8이라서, 클래스 0부터 8까지 총 9개 칸을 만듦


########### 원핫 2. pd.get_dummies ############
import pandas as pd

y = pd.get_dummies(x[0], dtype=int, drop_first=False)
print(y)
#    1  2  3  4  5  6  7  8
# 0  0  0  1  0  0  0  0  0
# 1  0  0  0  1  0  0  0  0
# 2  1  0  0  0  0  0  0  0
# 3  1  0  0  0  0  0  0  0
# 4  0  0  0  0  1  0  0  0
# 5  0  0  0  0  0  1  0  0
# 6  0  0  0  0  0  0  1  0
# 7  0  1  0  0  0  0  0  0
# 8  0  1  0  0  0  0  0  0
# 9  0  0  0  0  0  0  0  1
print(y.shape) # (10, 8)
# 토큰 1~8이 등장하므로 결과 크기는 (10, 8)
# drop_first=True는 첫 범주에 해당하는 열 하나를 빼므로 (10, 7)이 됨


########### 원핫 3. OneHotEncoder ############
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder(sparse_output=False) # sparse(희소행렬) 형태로 반환하지 않고 배열 그대로 반환

# OneHotEncoder는 입력을 (샘플 수, 특성 수)로 해석하므로 그대로 넣으면 샘플 1개, 특성 10개로 입력됨
# 토큰 각각을 샘플 하나로 다뤄서 토큰 값을 원-핫 인코딩하려면 입력을 (토큰 수, 1)로 바꿔야 함
x_2d = np.array(x).reshape(-1, 1) 

y = ohe.fit_transform(x_2d) 
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
print(y)
print(y.shape) # (10, 8)
# 토큰 1~8이 등장하므로 결과 크기는 (10, 8)