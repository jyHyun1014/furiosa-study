# 61-2 카피

# 정수 번호로 변환한 입력 문장들을 원핫인코딩 후 모델링

import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. 데이터
docs = [
    "너무 재미있다", "참 최고예요", "참 잘 만든 영화예요", "추천하고 싶은 영화입니다", "한 번 더 보고 싶어요", 
    "글쎄", "별로예요", "생각보다 지루해요", "연기가 어색해요", "재미없어요", 
    "너무 재미없다", "참 재밌네요", "개똥이 바보", "말똥이 잘생겼다", "길동이 또 거짓말한다",
]
labels = np.array([1,1,1,1,1, # 긍정:1, 부정:0 으로 라벨링
                   0,0,0,0,0,
                   0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, 
#  '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15, '글쎄': 16, '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
#  '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '개똥이': 25, '바보': 26, '말똥이': 27, '잘생겼다': 28, '길동이': 29, '또': 30, '거짓말한다': 31}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26], [27, 28], [29, 30, 31]]



######################### 패딩 Padding ############################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, 
                         padding="pre", # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                         maxlen=5,
                         truncating='post' # 자른다면 어디를 자를지
                         )
print(padded_x)
print(padded_x.shape) # (15, 5)


# 예측할 문자열
x_predict = ["개똥이 잘생겼다", "생각보다 재미있다", "말똥이 지루해요"]
x_predict = token.texts_to_sequences(x_predict)
print(x_predict)
# [[25, 28], [18, 3], [27, 19]]
x_predict = pad_sequences(x_predict,
                          padding="pre", # 문장 길이가 5보다 짧으면 앞쪽을 0으로 채움 # pre: 앞쪽 채우기, post: 뒷쪽 채우기
                          maxlen=5, # 모든 문장을 토큰 5개 길이로 맞춤
                          truncating='post' # 문장이 5개보다 길면 뒤쪽 토큰부터 제거                     
                          )
print(x_predict)
# [[ 0  0  0 25 28]
#  [ 0  0  0 18  3]
#  [ 0  0  0 27 19]]
print(x_predict.shape) # (3, 5)


# 원핫인코딩
from tensorflow.keras.utils import to_categorical
padded_x = to_categorical(padded_x) # 무조건 0부터 시작함
print(padded_x)
print(padded_x.shape) # (15, 5, 32) # (문장 수, 토큰 수, 클래스 수)

x_predict = to_categorical(x_predict)
print(x_predict.shape) # (3, 5, 29)


x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (12, 5, 32) (3, 5, 32) (12,) (3,)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(5, input_shape=(5, 32)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1, activation="sigmoid"))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

hist = model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.2,
    shuffle=True,
    )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("===========================")
print("loss :", loss[0]) #
print("acc :", round(loss[1], 4))
print("===========================")
# ===========================
# loss : 1.7016693353652954
# acc : 0.6667
# ===========================

y_pred = model.predict(x_test) # 0과 1사이의 실수 값으로 나옴
y_pred = np.round(y_pred) # 0 또는 1로 반올림
print(y_pred)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)

print(np.round(model.predict(x_predict)))
# [[1.]
#  [1.]
#  [1.]]
