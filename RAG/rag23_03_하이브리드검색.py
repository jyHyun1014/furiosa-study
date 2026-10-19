# 23-2 카피

# BM25 키워드 검색과 Chroma 벡터 검색을 결합한 하이브리드 검색

# EnsembleRetriever를 이용하여 두 검색 결과를 가중치 기반으로 결합
# BM25의 정확한 키워드 검색과 벡터 검색의 의미적 유사도 검색을 함께 활용

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

query = "삼성전자의 반도체 사업"

###################################### BM25 검색 ######################################
# BM25(Best Matching 25): 키워드 기반 문서 검색 알고리즘
# 검색어의 등장 빈도(TF), 역문서 빈도(IDF), 문서 길이 등을 고려하여 관련성이 높은 문서를 검색

from langchain_community.retrievers import BM25Retriever

# Document 목록을 기반으로 BM25 검색기 생성
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3


###################################### 벡터 검색 ######################################
# 벡터 검색: 텍스트를 임베딩 벡터로 변환하고 의미적 유사도를 기반으로 관련 문서 검색

import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# # Document를 임베딩하여 Chroma VectorDB에 저장 (최초 실행 시 사용)
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

# Chroma VectorStore를 Retriever로 변환
vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)


###################################### 하이브리드 검색 ######################################
# Hybrid Search: 서로 다른 검색 방식(BM25 + 벡터 검색)을 결합하는 검색 기법
# BM25는 키워드 일치 정도를, 벡터 검색은 의미적 유사도를 기반으로 문서 검색
# 두 검색 방식의 장점을 결합하여 검색 정확도와 재현율 개선을 기대할 수 있음

from langchain_classic.retrievers import EnsembleRetriever

# EnsembleRetriever: 여러 Retriever의 검색 결과를 결합하는 클래스
# 기본적으로 Weighted Reciprocal Rank Fusion(가중 RRF) 방식으로 결과를 통합
# 각 Retriever의 검색 순위와 가중치를 이용하여 문서별 통합 점수를 계산
# 동일한 문서는 중복 제거하고 통합 점수가 높은 순서로 정렬

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever, # 키워드 기반 검색기
        vector_retriever # 의미 기반 벡터 검색기
    ],
    weights=[0.5, 0.5], # 각 Retriever의 가중치 (BM25 50%, 벡터 검색 50%)
)

# BM25와 벡터 검색을 각각 수행한 후 RRF 방식으로 검색 결과 통합
hybrid_docs = hybrid_retriever.invoke(query)

for doc in hybrid_docs:
    print(doc.page_content)


# 하이브리드 검색(Hybrid Search)과 앙상블 검색(Ensemble Search)의 차이

# Hybrid Search: 서로 다른 검색 방식(키워드 검색 + 벡터 검색 등)을 결합하는 검색 기법. 검색 방식 간 장점 보완
# Ensemble Search: 여러 검색기의 결과를 결합하여 최종 검색 순위를 결정하는 기법. 검색 결과의 품질 향상

# EnsembleRetriever를 이용하여 하이브리드 검색 구현할 수 있음