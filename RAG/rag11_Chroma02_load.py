# 11-1 카피
# Chroma에 저장된 문서 벡터를 불러와 사용자 질문과 의미적으로 유사한 문서 청크를 검색

# pip install langchain-community
# pip install langchain-chroma

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"


# 임베딩은 텍스트를 의미 검색에 사용할 숫자 벡터로 변환함
# 벡터 DB에 저장할 때와 검색할 때 동일한 임베딩 모델을 사용해야 함
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# Chroma에 저장된 기존 벡터 DB를 불러옴
DB_PATH = "./_db/Chroma11/"
db = Chroma(
    embedding_function=embeddings, # 검색할 질문을 동일한 임베딩 모델로 벡터화
    persist_directory=DB_PATH, # Chroma 데이터가 저장된 로컬 경로
    collection_name="chroma11" # 사용할 Chroma 컬렉션 이름
)

# 저장된 데이터 확인
print("========================================")
print(db.get())
# {'ids': ['cda6fdfc-4772-49fc-bed0-7d2ab6302a49', '13392c88-8f53-4e5b-aa84-8ea9a68e1e84', ... 'a63ed541-20f9-49f4-894f-cc2a09a8607d', '7cb35580-ff87-4219-ac01-72ffe8052534'], 
# 'embeddings': None, 
# 'documents': ['삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전...', ... '게임용 GPU, 전문 시각화, 자동차 분야는 데이터센터 외의 사업 기반을 제공한다. 다만 인공지능 데이터센터...'], 
# 'uris': None, 
# 'included': ['metadatas', 'documents'], 
# 'data': None, 
# 'metadatas': [{'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/samsung_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}, {'source': './_data/rag_data/nvidia_outlook.txt'}]}

# db.get()은 컬렉션에 저장된 데이터를 반환함
# ids: 각 Document에 부여된 고유 ID
# documents: Chroma에 저장된 원본 텍스트
# metadatas: 각 Document에 함께 저장된 metadata
# embeddings: 저장된 임베딩 벡터
# 기본 설정에서는 embeddings가 반환되지 않음


print("========================================")
# 입력한 질문과 의미적으로 유사한 Document를 검색
# 질문도 embedding_function으로 벡터화한 후 저장된 벡터와 유사도를 비교함
# k=2이므로 유사도가 높은 Document 2개를 반환
aaa = db.similarity_search(
    "삼성전자 사업전망에 대해 알려줘", 
    k=2 # 반환할 Document의 개수 (기본값 4)
)

print(aaa)
# [Document(id='cda6fdfc-4772-49fc-bed0-7d2ab6302a49', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), 
#  Document(id='0781ccb9-47cb-4426-a6f4-6ca65e6df4b1', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')]

# 반환 결과는 Document 객체의 리스트
# id: Document의 고유 ID
# metadata: 원본 문서의 출처 등 부가 정보
# page_content: 검색된 Document의 실제 텍스트