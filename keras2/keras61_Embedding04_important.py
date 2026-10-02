# 61-3 카피

# 정수 번호로 변환한 입력 문장들을 원핫인코딩 하는 대신, 임베딩 벡터로 변환 후 모델링

import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
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
token.fit_on_texts(docs) # # 15문장 전체로 단어 사전을 만듦 -> 단어 31개 (번호 1 ~ 31)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, 
#  '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15, '글쎄': 16, '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
#  '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '개똥이': 25, '바보': 26, '말똥이': 27, '잘생겼다': 28, '길동이': 29, '또': 30, '거짓말한다': 31}

x = token.texts_to_sequences(docs) # # 위에서 만든 단어 사전으로 문장을 단어 번호 리스트로 바꿈 (문장 15개 -> 리스트 15개)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26], [27, 28], [29, 30, 31]]


######################### 패딩 Padding ############################
# 문장마다 단어 수가 다름 (1개 ~ 5개) -> 모델은 같은 길이만 받을 수 있어서 모자란 자리를 0 으로 채움
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, 
                         padding="pre",
                         maxlen=5,
                         truncating='post'
                         )
print(padded_x)
print(padded_x.shape) # (15, 5)




# 2. 모델 구성
from tensorflow.keras.layers import Embedding

model = Sequential()
# 단어 집합의 크기, 임베딩 벡터의 차원, max_len
# 고유한 단어의 수, 임베딩 벡터의 크기, 각 입력 시퀀스의 크기
# 단어 사전의 크기, 차원(단어 하나를 몇 칸짜리 벡터로 만들지), 입력 시퀀스 길이 (padding 한 길이)

# Embedding 파라미터 개수 = input_dim * output_dim
model.add(Embedding(input_dim=31+1, output_dim=100, input_length=5)) #  출력 (None, 5, 100) # 파라미터 개수 3200

# model.add(Embedding(input_dim=32, output_dim=100))   # input_length 생략 가능
# # [참고] input_length 를 안 쓰면 summary 의 Embedding Output Shape 가 (None, None, 100) 으로 나옴
# #        -> 문장 길이를 "아직 모름" 상태로 두고, 실제 데이터 (15, 5) 가 들어올 때 5 로 맞춰짐
# #        -> Param 수는 input_dim x output_dim 이라 input_length 와 상관없이 똑같이 3,200

# model.add(Embedding(32, 100))                 # input_dim, output_dim 이름 생략 가능 (순서 : 앞이 input_dim, 뒤가 output_dim)

# model.add(Embedding(32, 100, 5))              # 에러 -> ValueError: Could not interpret initializer identifier: 5
#                                               # 세 번째 자리는 input_length 가 아니라 embeddings_initializer (가중치 초기값 방법) 자리라서
#                                               # 5 를 초기값 방법 이름으로 읽으려다 실패함

# model.add(Embedding(32, 100, input_length=5)) # input_length 는 이름을 직접 써야 함


model.add(SimpleRNN(10)) # Embedding 출력 (None, 5, 100) 이 3차원이라 RNN 계열에 바로 넣을 수 있음 (reshape 필요 없음)
model.add(Dense(1, activation="sigmoid"))

model.summary()

######################### Embedding 적용 후 (SimpleRNN 없이 Embedding -> Dense) #########################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3200

#  dense (Dense)               (None, 5, 1)              101

# =================================================================
# Total params: 3,301
# Trainable params: 3,301
# Non-trainable params: 0
# _________________________________________________________________
# Embedding Param = input_dim x output_dim = 32 x 100 = 3,200 (bias 없음 -> 단어마다 100칸짜리 벡터 1줄씩, 32줄짜리 표)

# Embedding의 출력 (None, 5, 100)은 문장의 5개 단어 각각을 100차원 벡터로 바꾼 것
# 그 뒤 Dense(1)은 각 단어 위치마다 값 하나를 출력하므로 결과가 (None, 5, 1)이 됨
# 즉, 문장 하나당 분류 결과 하나가 아니라 단어마다 결과 5개를 냄
# 내가 원하는 건 문장당 예측 1개
############################### 요약: Dense의 입력이 2차원이어야 하기 때문에 RNN을 넣어줘야 한다!!!!! ###############################



######################### simpleRNN 적용 후 #########################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3200

#  simple_rnn (SimpleRNN)      (None, 10)                1110

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 4,321
# Trainable params: 4,321
# Non-trainable params: 0
# _________________________________________________________________
# SimpleRNN Param = units x (units + feature + 1) = 10 x (10 + 100 + 1) = 1,110 (feature = Embedding 의 output_dim 100)
# Dense Param = (10 + 1) x 1 = 11
# SimpleRNN 이 단어 5개를 순서대로 읽고 마지막 상태 1개 (None, 10) 만 출력 -> 문장 하나에 답 1개 (None, 1)


exit(  )
# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(padded_x, labels, epochs=100)








