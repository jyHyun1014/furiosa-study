# 17-1 카피

# 텍스트 파일을 청크로 나누고 임베딩한 뒤 FAISS에 저장하기

# 필요한 패키지
# pip install langchain-community
# pip install langchain-chroma
# pip install faiss-cpu

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# 3. 임베딩 모델 준비
# 임베딩은 텍스트를 의미 검색에 사용할 숫자 벡터로 변환함
# 벡터 DB에 저장할 때와 검색할 때 동일한 임베딩 모델을 사용해야 함
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# 4. 저장된 FAISS VectorStore 불러오기
DB_PATH = "./_db/Faiss17/"

# 로컬에 저장된 FAISS VectorStore를 불러옴
# 저장된 FAISS 인덱스와 Document 정보를 복원함
db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings, # 검색할 때 사용할 임베딩 모델
    allow_dangerous_deserialization=True, # pickle 데이터 역직렬화를 허용
)

print("=======================================================")
# FAISS 벡터와 Document ID의 매핑 정보 확인
print(db.index_to_docstore_id)
# {0: 'ffabbb28-412f-449f-b8b5-5b4da7c6311b', 
#  1: '7685d6f0-0725-4478-a116-468dd0b1145c', 
# ...
#  16: '5e289ca1-55f7-4011-a39b-318682258bf5', 
#  17: '48936ac6-8cfb-42f1-9b7d-a5dd376e4a8e'}

print("=======================================================")
# Document ID와 실제 Document 내용 확인
print(db.docstore._dict)
# {'ffabbb28-412f-449f-b8b5-5b4da7c6311b': Document(id='ffabbb28-412f-449f-b8b5-5b4da7c6311b', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), 
#  '7685d6f0-0725-4478-a116-468dd0b1145c': Document(id='7685d6f0-0725-4478-a116-468dd0b1145c', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='반도체 부문에서 주목할 변화는 인공지능 데이터센터의 확산이다. 대규모 언어 모델을 학습하고 서비스하려면 높은 연산 성능뿐 아니라 데이터를 빠르게 주고받는 메모리가 필요하다. 이에 따라 고대역폭 메모리와 서버용 D램에 대한 관심이 커지고 있다. 삼성전자가 제품 성능과 생산 수율을 개선하고 주요 고객사의 품질 검증을 통과한다면, 인공지능 인프라 투자 확대를 매출 성장으로 연결할 가능성이 있다.'), 
#  'fcda4f69-b877-4ac2-bb7d-7ffa7a1c9f98': Document(id='fcda4f69-b877-4ac2-bb7d-7ffa7a1c9f98', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='메모리 산업은 수요가 늘어도 실적 개선이 자동으로 보장되지는 않는다. 반도체 업체들이 생산을 크게 늘리면 공급이 수요를 앞질러 가격이 하락할 수 있다. 반대로 재고가 줄고 데이터센터 투자가 이어지면 제품 가격이 회복될 수 있다. 따라서 인공지능 시장의 성장률만 볼 것이 아니라 메모리 가격, 재고 수준, 생산량, 고객사의 실제 구매 상황을 함께 확인해야 한다. 고성능 제품의 비중이 높아지는지도 수익성을 판단하는 중요한 단서다.'), 
# ...

print("=======================================================")
# 질문을 임베딩한 후 FAISS에서 유사한 문서를 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)
# [Document(id='ffabbb28-412f-449f-b8b5-5b4da7c6311b', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), 
#  Document(id='4aaf2c3e-c032-4a26-b42f-270a7125c161', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')]