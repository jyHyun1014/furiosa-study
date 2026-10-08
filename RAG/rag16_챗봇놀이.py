# 15 카피

# Chroma에 저장된 문서 벡터를 불러와 사용자 질문과 의미적으로 유사한 문서 청크를 검색하고 LLM에 전달하여 답변을 생성한 뒤 Gradio 챗봇을 통해 사용자에게 제공
# k 값을 Gradio에서 직접 조절하기 + context 검색 결과를 Gradio에서 같이 보여주기

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

# 기존에 저장된 Chroma Vector Store 불러오기
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12"
)


# Chroma에 저장된 문서(chunk)의 개수 확인
print("================================================")
print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}") # 벡터 저장소에 저장된 문서 수 : 59

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
# Gradio 질문 처리 함수
# ============================================================
import gradio as gr

def ask_rag(query, k):

    # 사용자가 선택한 k값으로 Retriever 생성
    retriever = vector_store.as_retriever(
        search_kwargs={"k": int(k)}
    )

    # 검색된 여러 Document를 context로 합친 후
    # Prompt에 전달하고 LLM을 통해 답변을 생성하는 Chain
    doc_chain = create_stuff_documents_chain(
        model,
        prompt,
    )

    # Retriever로 관련 문서를 검색하고
    # 검색된 문서를 context로 전달하여 doc_chain 실행
    rag_chain = create_retrieval_chain(
        retriever,
        doc_chain
    )

    # RAG Chain 실행
    response = rag_chain.invoke({
        "input": query
    })

    # 검색된 Document
    context = response["context"]

    # 검색된 Document를 화면에 표시할 문자열 생성
    documents_text = ""

    for i, doc in enumerate(context):
        documents_text += f"===== Document {i + 1} =====\n"
        documents_text += doc.page_content
        documents_text += "\n\n"

    # LLM 답변과 검색된 문서를 반환
    return response["answer"], documents_text


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# Chroma RAG 챗봇")

    # 질문 입력
    query = gr.Textbox(
        label="질문",
        placeholder="질문을 입력하세요."
    )

    # 검색할 Document 개수
    k = gr.Slider(
        minimum=1,
        maximum=10,
        value=2,
        step=1,
        label="검색할 문서 수 (k)"
    )

    # 질문 실행 버튼
    submit = gr.Button("질문하기")

    # LLM 답변
    answer = gr.Textbox(
        label="답변",
        lines=5
    )

    # Retriever가 검색한 Document
    documents = gr.Textbox(
        label="검색된 문서",
        lines=15
    )

    # 버튼 클릭 → ask_rag 실행
    submit.click(
        fn=ask_rag,
        inputs=[query, k],
        outputs=[answer, documents]
    )


# ============================================================
# Gradio 실행
# ============================================================

demo.launch()