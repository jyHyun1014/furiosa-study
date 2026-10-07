# 11-1 카피

# 텍스트 문서를 청크로 분할하고 임베딩하여 Chroma에 저장한 후, 질문과 관련된 문서 청크를 검색

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
# 1. 원본 문서 불러오기
# ============================================================

path = "./_data/rag_data/"

# 폴더에서 .txt 파일 목록을 가져옴
txt_files = glob(os.path.join(path, "*.txt"))
print(txt_files) # ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']

# 각 텍스트 파일을 Document로 변환하여 하나의 리스트에 저장
data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding="utf-8")
    data += loader.load()

# 불러온 Document 확인
print("================================================")
print("불러온 문서 수 :", len(data)) # 불러온 문서 수 : 3
print(data) 
# [Document(metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}, page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산\n\n작성 ...재작성된 학습 자료이다.\n'), 
# Document(metadata={'source': './_data/rag_data\\nvidia_outlook.txt'}, page_content='엔비디아 사업 전망\n\n엔비디아는 그래픽처리장치와 이를 활용하는 ...투자 권유가 아니다.\n'), 
# Document(metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스... 자료이며 투자 권유가 아니다.\n')]

# 첫 번째 문서의 실제 텍스트 확인
print("================================================")
print(data[0].page_content)
# 2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산
# ...
# 이 문서는 기사 원문이 아니라 LangChain 교육 및 RAG 실습을 목적으로 재작성된 학습 자료이다.

# 각 문서의 문자 수 확인
print("================================================")
char_count = [len(doc.page_content) for doc in data] # page_content에 저장된 텍스트의 길이를 계산
print(char_count) # [8158, 2049, 1898]

# ============================================================
# 2. 문서를 Chunk로 분할
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 하나의 chunk가 가질 최대 길이
    chunk_overlap=100, # 인접한 chunk 사이에 겹치는 최대 길이
    separators=["\n\n", "\n", " ", ""], # 문단 → 줄 → 공백 → 문자 순으로 분할
)

# 각 Document를 여러 개의 작은 Document(chunk)로 분할
# 원본 Document의 metadata는 분할된 chunk에도 유지됨
texts = text_splitter.split_documents(data)

print("================================================")
print("생성된 텍스트 청크 수 :", len(texts)) # 생성된 텍스트 청크 수 : 59
print("각 청크의 길이 :", list(len(text.page_content) for text in texts))
# 각 청크의 길이 : [259, 154, 150, ..., 170, 187, 249]

# 첫 번째 chunk 확인
print("================================================")
print("첫번째 청크의:", texts[0])
# 첫번째 청크의 내용 : page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산
# \n
# 작성 목적
# 이 문서는 2026년 10월 초 공개된 국내 인공지능 관련 최신 보도를 바탕으로 LangChain의 문서 로딩, 텍스트 분할, 임베딩, 벡터 데이터베이스 저장, 검색 및 RAG 실습에 활용할 수 있도록 재구성한 학습용 텍스트이다. 특정 언론사의 기사 원문을 복제하지 않고 공개된 사실과 기술적 배경을 중심으로 내용을 확장해 작성하였다.
# \n
# 1. 한국의 AI for All 프로젝트' metadata={'source': './_data/rag_data\\2026_AI_for_All.txt'}
print("첫번째 청크의 길이 :", len(texts[0].page_content))

# 두 번째 chunk 확인
print("두번째 청크의 내용 :",  texts[1])

# ============================================================
# 3. 임베딩 모델 준비
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
# 4. 청크를 임베딩하여 Chroma에 저장
# ============================================================

DB_PATH = "./_db/Chroma12/"

# 각 chunk를 임베딩하여 벡터로 변환한 후 Chroma에 저장
# Document의 page_content와 metadata도 함께 저장됨
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12"
)

# Chroma에 저장된 문서(chunk)의 개수 확인
print("================================================")
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}") # 벡터 저장소에 저장된 문서 수 : 59

# ============================================================
# 5. Vector Store에서 직접 유사도검색
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
# 6. Vector Store를 Retriever로 변환
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