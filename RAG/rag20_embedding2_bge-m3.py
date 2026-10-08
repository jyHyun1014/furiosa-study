# 10-1 카피

# https://huggingface.co/BAAI/bge-m3

# pip install langchain_huggingface
# pip install sentence-transformers

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'

# Hugging Face의 BGE-M3 임베딩 모델 생성
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3', # 고성능 다국어 텍스트 임베딩 모델
    model_kwargs={
        "device": "cpu", # CPU에서 모델 실행
        # "local_files_only": True, # 로컬에 다운로드된 모델만 사용
    },
)

vector = embeddings.embed_query(prompt)            

print(vector)
print('==============================')

print('임베딩 벡터의 차원 :', len(vector)) # 임베딩 벡터의 차원 : 1024