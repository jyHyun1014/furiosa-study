# 11-1 카피

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

# 1. 원본 문서 불러오기
data_path = "./_data/rag_data/"
# TextLoader를 사용하여 텍스트 파일을 Document 형태로 불러옴
loader1 = TextLoader(data_path + "samsung_outlook.txt", encoding="utf-8")
loader2 = TextLoader(data_path + "nvidia_outlook.txt", encoding="utf-8")

# 2. 문서를 검색하기 좋은 작은 조각(chunk)으로 분할

# 여러 개의 구분자(separator)를 우선순위에 따라 재귀적으로 적용하면서 텍스트를 나눔
# 우선 문단(\n\n)으로 나눠보고 그 결과 중 chunk_size보다 큰 부분에 대해서만 줄(\n), 공백, 문자 단위 순서로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 하나의 chunk가 가질 최대 문자 수
    chunk_overlap=100, # 인접한 chunk가 서로 겹쳐서 포함할 문자 수
    separators=["\n\n", "\n", " ", ""],  # 우선순위대로 적용할 분할 기준
)

# 각 loader는 파일을 Document로 읽고, splitter를 적용해 여러 Document로 만듦
# 원본 파일 경로는 Document의 metadata['source']에 저장됨
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

print(split_doc1)
# [Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='반도체 부문에서 주목할 변화는 인공지능 데이터센터의 확산이다. 대규모 언어 모델을 학습하고 서비스하려면 높은 연산 성능뿐 아니라 데이터를 빠르게 주고받는 메모리가 필요하다. 이에 따라 고대역폭 메모리와 서버용 D램에 대한 관심이 커지고 있다. 삼성전자가 제품 성능과 생산 수율을 개선하고 주요 고객사의 품질 검증을 통과한다면, 인공지능 인프라 투자 확대를 매출 성장으로 연결할 가능성이 있다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='메모리 산업은 수요가 늘어도 실적 개선이 자동으로 보장되지는 않는다. 반도체 업체들이 생산을 크게 늘리면 공급이 수요를 앞질러 가격이 하락할 수 있다. 반대로 재고가 줄고 데이터센터 투자가 이어지면 제품 가격이 회복될 수 있다. 따라서 인공지능 시장의 성장률만 볼 것이 아니라 메모리 가격, 재고 수준, 생산량, 고객사의 실제 구매 상황을 함께 확인해야 한다. 고성능 제품의 비중이 높아지는지도 수익성을 판단하는 중요한 단서다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='파운드리는 고객사가 설계한 반도체를 위탁 생산하는 사업이다. 삼성전자는 첨단 제조 공정과 반도체 설계 지원 역량을 바탕으로 고객을 확보하려 한다. 이 시장에서는 공정의 미세화뿐 아니라 실제 생산에서 불량을 줄이는 수율, 안정적인 납기, 고객사의 설계 일정에 맞춘 지원이 중요하다. 수주가 발표되더라도 제품 양산과 충분한 수율 확보까지는 시간이 걸릴 수 있으므로, 계약 소식과 이익 기여 시점을 나누어 살펴볼 필요가 있다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='스마트폰 사업은 프리미엄 제품의 교체 수요와 중저가 제품의 판매량이 함께 영향을 미친다. 삼성전자는 제품 자체의 완성도뿐 아니라 운영체제 업데이트, 보안 지원, 기기 간 연결 경험을 통해 기존 고객을 유지하려 한다. 다양한 가격대의 제품을 공급하면 폭넓은 소비자층에 접근할 수 있지만, 제품군이 지나치게 복잡해지면 마케팅과 재고 관리 부담이 커질 수 있다. 고급형 제품에서는 카메라, 화면, 인공지능 기능, 운영체제와 서비스의 연결성이 차별화 요소가 된다. 인공지능 기능은 사진 편집이나 정보 검색처럼 소비자가 자주 사용하는 작업에서'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='고급형 제품에서는 카메라, 화면, 인공지능 기능, 운영체제와 서비스의 연결성이 차별화 요소가 된다. 인공지능 기능은 사진 편집이나 정보 검색처럼 소비자가 자주 사용하는 작업에서 편리함을 제공할 때 제품 선택에 영향을 줄 수 있다. 기능을 탑재하는 것만으로는 충분하지 않으며, 개인정보 보호와 처리 속도, 지원 언어와 이용 비용도 사용자 경험을 좌우한다. 제품 기능이 개선되더라도 소비자가 새 기기로 바꿀 이유가 부족하면 판매 증가가 제한될 수 있다. 경기 불확실성이 커지면 스마트폰 교체 주기가 길어지는 경향이 생길 수 있어, 신제품의'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='기능이 개선되더라도 소비자가 새 기기로 바꿀 이유가 부족하면 판매 증가가 제한될 수 있다. 경기 불확실성이 커지면 스마트폰 교체 주기가 길어지는 경향이 생길 수 있어, 신제품의 혁신 수준과 실제 체감 가치가 더욱 중요해진다. 부품 가격, 마케팅 비용, 환율과 경쟁사의 신제품 출시도 수익성에 영향을 준다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='가전과 디스플레이 사업은 소비 심리와 교체 주기의 영향을 받는다. 경기가 불확실하면 소비자는 TV나 냉장고 같은 고가 제품의 구매를 미룰 수 있다. 에너지 효율, 제품 디자인, 스마트홈 연동, 프리미엄 모델 판매 비중은 경쟁력을 높이는 요소다. 그러나 여러 제조사가 비슷한 기능을 제공하면 가격 경쟁이 심해져 이익률이 낮아질 수 있다.'), 
#  Document(metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')
#  ]

print(f"삼성 문서 청크 수: {len(split_doc1)}") # 9
print(f"엔비디아 문서 청크 수: {len(split_doc2)}") # 9

# 3. 임베딩 모델 준비
# 임베딩은 텍스트를 의미 검색에 사용할 숫자 벡터로 변환함
# 벡터 DB에 저장할 때와 검색할 때 동일한 임베딩 모델을 사용해야 함
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# 4. FAISS 인덱스 생성
embedding_dim = len(embeddings.embed_query("hello world")) # 임베딩 벡터의 차원 수를 확인
faiss_index = faiss.IndexFlatL2(embedding_dim) # 벡터 간 L2(제곱 유클리드) 거리를 계산하여 가장 가까운 벡터를 검색하는 FAISS 인덱스
# faiss_index = faiss.IndexFlatL2(1536) # 차원을 알고 있다면 직접 지정 가능
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 인덱스의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d) # 1536

# 5. FAISS VectorStore 생성
# FAISS 인덱스와 Document를 연결하여 LangChain의 FAISS VectorStore 생성
faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(), # Document를 메모리에 저장
    index_to_docstore_id={}, # FAISS 벡터와 Document ID를 연결하는 매핑
)
# 현재 저장된 벡터 개수 확인
print(faiss_db.index.ntotal) # 0

# 6. 문서를 임베딩하여 FAISS에 저장
db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

# 7. FAISS VectorStore를 로컬에 저장
# FAISS 인덱스와 Document 정보를 로컬에 저장
# 프로그램을 종료해도 저장된 FAISS DB를 다시 불러올 수 있음
DB_PATH = "./_db/Faiss17/"
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)
