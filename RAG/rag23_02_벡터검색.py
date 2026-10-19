# 23-1 카피

# BM25 키워드 검색과 Chroma 벡터 검색의 결과를 비교하기

# BM25: 검색어와 문서의 키워드 일치 정도를 기반으로 검색
# 벡터 검색: 텍스트를 임베딩 벡터로 변환하고 의미적 유사도를 기반으로 검색

# pip install rank_bm25

from langchain_core.documents import Document

# 검색 대상 문서 생성
texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

# 검색할 질문
query = "삼성전자의 반도체 사업"

###################################### BM25 검색 ######################################
# BM25(Best Matching 25): 키워드 기반 문서 검색 알고리즘
# 검색어의 등장 빈도(TF), 역문서 빈도(IDF), 문서 길이 등을 고려하여 관련성 점수 계산
# 기본적으로 공백 기준 토큰화를 사용하므로 한국어 조사 차이에 따른 검색 한계가 있음

from langchain_community.retrievers import BM25Retriever

# Document 목록을 기반으로 BM25 검색기 생성
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

bm25_docs = bm25_retriever.invoke(query)

print("==================== BM25 검색 결과 ====================")
for doc in bm25_docs:
    print(doc.page_content)

###################################### 벡터 검색 ######################################
# 벡터 검색: 텍스트를 임베딩 벡터로 변환하고 벡터 간 유사도를 기반으로 관련 문서 검색
# 키워드가 정확히 일치하지 않아도 의미적으로 유사한 문서를 검색할 수 있음

import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# 텍스트를 숫자 벡터로 변환함
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# # 임베딩하여 벡터로 변환한 후 Chroma에 저장
# DB_PATH = "./_db/Chroma23/"
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_PATH,
#     collection_name="chroma23"
# )

# Chroma에 저장된 임베딩 벡터 불러오기
DB_PATH = "./_db/Chroma23/"
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma23"
)

print(f"Chroma에 저장된 문서 수 : {vector_store._collection.count()}") # Chroma에 저장된 문서 수 : 8

vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)
vector_docs = vector_retriever.invoke(query)

print("==================== VectorDB 검색 결과 ====================")
for doc in vector_docs:
    print(doc.page_content)


# ==================== BM25 검색 결과 ====================
# 삼성전자는 AI 반도체 기술을 연구한다.
# 반도체는 컴퓨터의 핵심 부품이다.
# 테슬라는 전기자동차를 제조한다.

# ==================== VectorDB 검색 결과 ====================
# 삼성전자는 반도체를 생산하는 기업이다.
# 삼성전자는 AI 반도체 기술을 연구한다.
# 삼성전자는 갤럭시 스마트폰을 판매한다.



# BM25 결과에 관련성이 낮은 테슬라 문서가 포함된 이유는 한국어 조사 때문에 키워드가 정확히 일치하지 않고, 일부 문서의 BM25 점수가 같거나 낮아 순위 구분이 어려워졌기 때문일 수 있음