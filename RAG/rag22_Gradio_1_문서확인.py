# 21 카피

# Chroma에서 관련 문서를 검색하고 이전 대화 기록을 반영하여 답변하는 RAG 챗봇을 LangChain과 Gradio로 구현하고, 실제 검색된 문서의 출처와 내용을 터미널에 출력

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
# DB_PATH = "./_db/Chroma21/"
# vector_store = Chroma.from_documents(
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=DB_PATH, # Chroma의 데이터를 로컬에 저장할 경로
#     collection_name="chroma21" # Chroma 저장소 내 컬렉션을 구분하기 위한 이름
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

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Prompt 설정
# ChatPromptTemplate: System, Human 등의 메시지를 조합하여 채팅 프롬프트 생성
# {context}: Retriever가 검색한 문서 내용
# {input}: 사용자가 현재 입력한 질문
# {chat_history}: 이전 사용자 질문과 AI 답변 기록
prompt = ChatPromptTemplate.from_messages([
    ("system", """다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
    "주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요.
    
    컨텍스트: {context}
    """),

    # 이전 대화 기록을 프롬프트에 삽입
    # HumanMessage와 AIMessage로 구성된 메시지 리스트가 전달됨
    MessagesPlaceholder(variable_name='chat_history'),

    # 현재 사용자가 입력한 질문
    ("human", "{input}"),
])

# RAG Chain 생성 
# 검색된 여러 Document를 하나의 context로 합친 후 prompt에 전달하고 LLM을 통해 답변을 생성하는 Chain
doc_chain = create_stuff_documents_chain( # prompt | model
    model, 
    prompt,
    # document_variable_name="context", # 검색된 Document를 전달할 Prompt 변수명 # 기본값이 "context"이므로 생략 가능
)
# Retriever와 doc_chain을 연결하여 RAG Chain 생성
# 1. Retriever가 사용자 질문(input)을 임베딩하여 관련 문서를 검색
# 2. 검색된 Document를 context에 전달
# 3. doc_chain이 context, chat_history, input을 이용하여 답변 생성
rag_chain = create_retrieval_chain(retriever, doc_chain)



# Gradio에서 사용자 질문을 처리하는 함수

# message: 현재 사용자가 입력한 질문 (str)
# history: 이전 대화 기록 (list)
# Gradio ChatInterface가 두 인자를 자동으로 전달함
def answer_invoke(message, history):
    chat_history =[] # LangChain에 전달할 대화 기록 리스트
    print("history: ", history)

    # 이전 대화 기록 중 최근 10개 메시지만 사용 # 최근 10턴이 아니라 사용자/AI 메시지를 합쳐 최대 10개
    for item in history[-10:]:
        print("item :", item)
        content = item["content"] # 각 메시지의 텍스트 내용 추출

        if item["role"] == "user": # 사용자 메시지를 LangChain의 HumanMessage로 변환
            chat_history.append(HumanMessage(content=content))
        elif item["role"] == "assistant": # AI 답변을 LangChain의 AIMessage로 변환
            chat_history.append(AIMessage(content=content))

    # RAG Chain 실행
    # input: 현재 사용자의 질문
    # chat_history: 이전 대화 기록
    # Retriever로 관련 문서를 검색한 후 LLM이 답변을 생성함
    response = rag_chain.invoke({
        "input": message,
        "chat_history": chat_history,
    })

    # RAG Chain이 실제로 검색한 문서 2개를 터미널에 출력하여 확인
    # response["context"]: Retriever가 검색한 Document 객체 리스트
    for i, doc in enumerate(response["context"], start=1): # # enumerate(start=1): 문서 번호를 1부터 부여
        print(f"\n검색 문서 {i}")

        # doc.metadata: 문서의 메타데이터(출처, 페이지 번호 등)
        # get("source"): 문서의 출처를 가져옴 (없으면 None 반환)
        print("출처 :", doc.metadata.get("source"))

        # doc.page_content: 검색된 Document의 실제 텍스트 내용
        print("내용 :", doc.page_content)

        print("=============================================")


    return response['answer'] # Gradio 채팅창에 표시할 최종 답변 반환

# Gradio 채팅 인터페이스 생성
# 사용자가 질문을 입력하면 answer_invoke()를 실행
# 이전 대화 기록(history)은 Gradio가 자동으로 관리하여 전달
demo = gr.ChatInterface(fn=answer_invoke, title='ㅋㅋ봇!!')

# Gradio 웹 서버 실행
demo.launch()


"""
1. 내 이름은 영선이야
history = []

2. 나는 누구인가?
history = [{'role': 'user', 'metadata': None, 'content': [{'text': '내 이름은 영선이야', 'type': 'text'}], 'options': None}, {'role': 'assistant', 'metadata': None, 'content': [{'text': '알겠어요, 영선님.', 'type': 'text'}], 'options': None}]
item1 = {'role': 'user', 'metadata': None, 'content': [{'text': '내 이름은 영선이야', 'type': 'text'}], 'options': None}
item2 = {'role': 'assistant', 'metadata': None, 'content': [{'text': '알겠어요, 영선님.', 'type': 'text'}], 'options': None}

"""