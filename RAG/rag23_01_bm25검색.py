
# BM25(Best Matching 25)를 이용한 키워드 기반 문서 검색

# BM25: 검색어와 문서에 포함된 단어의 일치 정도를 기반으로 관련성 점수를 계산하는 검색 알고리즘
# 검색어의 등장 빈도(TF), 역문서 빈도(IDF), 문서 길이 등을 고려하여 관련성이 높은 문서를 검색
# 임베딩을 사용하는 벡터 검색과 달리 단어의 의미적 유사도보다 키워드 일치에 중점을 둠

# pip install rank_bm25

from langchain_core.documents import Document

# 검색 대상 문서 생성
# Document: 텍스트(page_content)와 메타데이터(metadata)를 저장하는 LangChain 문서 객체
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

# BM25Retriever: BM25 알고리즘을 이용하여 키워드 기반으로 문서를 검색하는 Retriever
from langchain_community.retrievers import BM25Retriever

# from_documents(): Document 목록을 기반으로 BM25 검색기 생성
# 내부적으로 문서를 토큰화하고 BM25 검색에 필요한 통계 정보를 계산
bm25_retriever = BM25Retriever.from_documents(texts)

# 검색 결과로 반환할 문서 개수 설정 (Top-k)
bm25_retriever.k = 3


# invoke(): 질문을 입력받아 BM25 점수가 높은 상위 3개 문서 검색
# 검색어와 문서의 키워드 일치 정도를 기반으로 관련성 점수를 계산
# 반환값: 관련성이 높은 순서로 정렬된 Document 객체 리스트
bm25_docs = bm25_retriever.invoke(query)

print("==================== BM25 검색 결과 ====================")
for doc in bm25_docs:
    print(doc.page_content)


## 추가로 알아둘 주의할 점
# BM25Retriever.from_documents()는 기본적으로 공백 기준으로 텍스트를 분리함
# ['삼성전자의', '반도체', '사업']
# 하지만 문서에는 '삼성전자는', '반도체를'처럼 조사가 붙어 있기 때문에 '삼성전자의'와 정확히 일치하지 않음
# 따라서 한국어 문서 검색에서는 형태소 분석기를 사용하는 것이 더 효과적이다