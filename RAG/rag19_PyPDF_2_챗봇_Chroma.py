import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import gradio as gr

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"


# # PDF 논문을 하나의 Document로 불러오기
# path = "./_data/"
# pdf_loader = PyPDFLoader(path + "1706.03762v7.pdf", mode="single")
# pdf_docs = pdf_loader.load()

# print("불러온 문서 수 :", len(pdf_docs)) # 불러온 문서 수 : 1
# # print(pdf_docs)

# print("각 문서의 문자 수 :", [len(doc.page_content) for doc in pdf_docs]) # [39524]

# # 문서를 Chunk로 분할
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, # 하나의 chunk가 가질 최대 길이
#     chunk_overlap=150, # 인접한 chunk 사이에 겹치는 최대 길이
#     separators=["\n\n", "\n", " ", ""], # 문단 → 줄 → 공백 → 문자 순으로 분할
# )

# # 각 Document를 여러 개의 작은 Document(chunk)로 분할
# texts = text_splitter.split_documents(pdf_docs)

# print("================================================")
# print("생성된 텍스트 청크 수 :", len(texts)) # 생성된 텍스트 청크 수 : 47
# print("각 청크의 길이 :", list(len(text.page_content) for text in texts)) # 각 청크의 길이 : [984, 944, ..., 998, 464]


# 텍스트를 숫자 벡터로 변환함
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', # 텍스트를 1536차원 벡터로 변환
    api_key=api_key,
    base_url=base_url,
)

# # Chroma에 임베딩 벡터 저장 # Document의 page_content와 metadata도 함께 저장됨
# DB_PATH = "./_db/Chroma19/"
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_PATH, # Chroma의 데이터를 로컬에 저장할 경로
#     collection_name="chroma19" # Chroma 저장소 내 컬렉션을 구분하기 위한 이름
# )
# print(f"Chroma에 저장된 문서 수 : {vector_store._collection.count()}") # Chroma에 저장된 문서 수 : 47

####################################################################################################################

# Chroma에 저장된 임베딩 벡터 불러오기
DB_PATH = "./_db/Chroma19/"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma19"
)
print(f"Chroma에 저장된 문서 수 : {vector_store._collection.count()}") # Chroma에 저장된 문서 수 : 47


# 벡터 저장소를 LangChain의 Retriever 객체로 변환
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2} # 검색 결과로 최대 2개의 Document를 반환
)
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000002453EDFE810> search_kwargs={'k': 2}

model = ChatOpenAI(
    model = "gpt-6-luna",
    temperature=0, # 응답의 무작위성을 낮춤
    max_tokens=1000, # 최대 생성 토큰 수
    api_key=api_key,
    base_url=base_url,
)

# Prompt 설정
# {context}에는 검색된 Document들이 들어가고 {input}에는 사용자의 질문이 들어감
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요.

컨텍스트: {context}

질문: {input}

답변:
""")

# RAG Chain 생성 # 검색된 여러 Document를 하나의 context로 합친 후 prompt에 전달하고 LLM을 통해 답변을 생성하는 Chain
doc_chain = create_stuff_documents_chain( # prompt | model
    model, 
    prompt,
    # document_variable_name="context", # 검색된 Document를 전달할 Prompt 변수명 # 기본값이 "context"이므로 생략 가능
)
rag_chain = create_retrieval_chain(retriever, doc_chain) # 문서검색 | prompt | model # Retriever로 관련 문서를 검색하고, 검색된 문서를 context라는 이름으로 전달하여 doc_chain 실행



def answer_invoke(message, history):
    response = rag_chain.invoke({"input": message})
    return response['answer']

# gradio 인터패이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='ㅋㅋ봇!!')

# Grdaio 실행
demo.launch()