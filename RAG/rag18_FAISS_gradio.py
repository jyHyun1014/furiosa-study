# 14 카피

# Chroma에 저장된 문서 벡터를 불러와 사용자 질문과 의미적으로 유사한 문서 청크를 검색하고 LLM에 전달하여 답변을 생성한 뒤 Gradio 챗봇을 통해 사용자에게 제공

# pip install langchain-community
# pip install langchain-chroma

import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.vectorstores import FAISS


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

# ============================================================
# Chroma에 저장된 문서를 불러오기
# ============================================================

DB_PATH = "./_db/Faiss17/"

# 기존에 저장된 Chroma Vector Store 불러오기

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings, # 검색할 때 사용할 임베딩 모델
    allow_dangerous_deserialization=True, # pickle 데이터 역직렬화를 허용
)


# FAISS 인덱스에 저장된 벡터 개수 확인
print("================================================")
print(f"벡터 저장소에 저장된 문서 수 : {vector_store.index.ntotal}") # 벡터 저장소에 저장된 문서 수 : 59

# ============================================================
# Vector Store를 Retriever로 변환
# ============================================================

# Vector Store를 LangChain의 Retriever 객체로 변환
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2} # 검색 결과로 최대 2개의 Document를 반환
)
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000025EDD223750> search_kwargs={'k': 2}

# ============================================================
# LLM 연결
# ============================================================
print("================================================")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = "gpt-6-luna",
    temperature=0, # 응답의 무작위성을 낮춤
    max_tokens=1000, # 최대 생성 토큰 수
    api_key=api_key,
    base_url=base_url,
)

# ============================================================
# Prompt 설정
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# {context}에는 검색된 Document들이 들어가고
# {input}에는 사용자의 질문이 들어감
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요.

컨텍스트: {context}

질문: {input}

답변:
""")


# ============================================================
# RAG Chain 생성
# ============================================================

# 검색된 여러 Document를 하나의 context로 합친 후 prompt에 전달하고 LLM을 통해 답변을 생성하는 Chain
doc_chain = create_stuff_documents_chain( # prompt | model
    model, 
    prompt,
    # document_variable_name="context", # 검색된 Document를 전달할 Prompt 변수명 # 기본값이 "context"이므로 생략 가능
)

# Retriever로 관련 문서를 검색한 후 검색된 Document를 doc_chain에 전달하여 답변 생성
rag_chain = create_retrieval_chain(retriever, doc_chain) # 문서검색 | prompt | model ## Retriever로 관련 문서를 검색하고, 검색된 문서를 context라는 이름으로 전달하여 doc_chain 실행

# # ============================================================
# # RAG Chain 실행
# # ============================================================

# # input만 전달하면 create_retrieval_chain이 질문을 Retriever에 전달하여 관련 Document를 검색하고 검색 결과를 context로 만들어 doc_chain에 전달함
# query = "삼성전자의 창업자는 누구인가요?"
# response = rag_chain.invoke({"input": query})

# print(response)
# # {'input': '삼성전자의 창업자는 누구인가요?', 'context': [Document(id='6a6cff9b-34a6-4f02-84bf-c478baf37395', metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), Document(id='b3866414-641c-4ec3-8d55-4a1549ac9247', metadata={'source': './_data/rag_data\\samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')], 'answer': '삼성전자의 창업자는 이병철(이건희의 선조)입니다. 삼성그룹의 창립자인 이병철 회장이 1938년에 삼성상회를 세우면서 삼성의 역사가 시작되었습니다. 이후 삼성전자는 삼성그룹의 주요 계열사로 발전했습니다.'}

# print("====================== keys() ==========================")
# print(response.keys())
# # dict_keys(['input', 'context', 'answer'])

# print("====================== context ==========================")
# print(response['context'][0].page_content)
# # 삼성전자 사업 전망
# # 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.

# print("======================= answer =========================")
# print(response['answer'])
# # 주어진 정보로는 답변할 수 없습니다.

# ============================================================
# Gradio 챗봇
# ============================================================
# pip install gradio
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input": message})
    return response['answer']

# gradio 인터패이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='영선봇!!')

# Grdaio 실행
demo.launch()