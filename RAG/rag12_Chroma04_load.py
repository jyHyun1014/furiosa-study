# 12-3 카피

# Chroma에 저장된 문서 벡터를 불러와 사용자 질문과 의미적으로 유사한 문서 청크를 검색

# pip install langchain-community
# pip install langchain-chroma

import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma


load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"


# ============================================================
# 임베딩 모델 준비
# ============================================================

# 임베딩은 텍스트를 의미 검색에 사용할 숫자 벡터로 변환함
# 벡터 DB에 저장할 때와 검색할 때 동일한 임베딩 모델을 사용해야 함
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# 임베딩 모델이 실제로 텍스트를 벡터로 변환하는지 확인
sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
print(vector) # [0.0367431640625, -0.031951904296875, ...
print(len(vector)) # 1536


# ============================================================
# Chroma에 저장된 문서를 불러오기
# ============================================================

DB_PATH = "./_db/Chroma12/"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12"
)


# Chroma에 저장된 Document(chunk)의 개수 확인
print("================================================")
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}") # 벡터 저장소에 저장된 문서 수 : 59

# ============================================================
# Vector Store에서 직접 유사도검색
# ============================================================

query = "삼성전자의 창업자는 누구인가요?"

# 질문을 임베딩한 후 저장된 벡터와 유사도를 비교하여 관련성이 높은 Document를 검색
# k를 지정하지 않으면 기본적으로 4개의 Document를 반환
result = vector_store.similarity_search(query)
print(f"검색 결과의 길이 : {len(result)}") # 검색 결과의 길이 : 4

# 검색된 Document 확인
print("첫 번째 검색 결과:")
print(result[0])
# page_content='삼성전자 사업 전망
# \n
# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.' 
# metadata={'source': './_data/rag_data\\samsung_outlook.txt'}


# ============================================================
# Vector Store를 Retriever로 변환
# ============================================================

# Vector Store를 LangChain의 Retriever 객체로 변환
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2} # 검색 시 관련 문서 2개를 반환
)
print("================================================")
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000025EDD223750> search_kwargs={'k': 2}

# Retriever를 통해 질문과 관련된 문서를 검색
# invoke()에 질문을 전달하면 관련 Document를 반환
retrieved_docs = retriever.invoke(query)

print(f"검색된 관련 문서 수 : {len(retrieved_docs)}") # 검색된 관련 문서 수 : 2
print(f"첫번째 관련 문서 내용 미리보기 : {retrieved_docs[0].page_content[:50]}...")
# 첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망
# \n
# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사...


###################################
# similarity_search()
# → Vector Store의 검색 기능을 직접 사용

# as_retriever()
# → Vector Store를 Retriever로 만들어 사용
# → 이후 RAG 체인 등에 연결하기 편함