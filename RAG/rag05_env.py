# .env 파일에서 설정한 API KEY 값 불러오기

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    # openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?


#############################################

# 방법1
from dotenv import load_dotenv
load_dotenv() # 현재 폴더(또는 상위 폴더)에서 .env 파일을 찾아 그 안의 값을 프로세스 환경변수로 불러옴
key = os.getenv("OPENAI_API_KEY") # 현재 프로세스의 환경변수에서 키를 읽음. 변수가 없으면 오류 대신 None을 반환

# 방법2
from dotenv import load_dotenv
load_dotenv() # 현재 폴더(또는 상위 폴더)에서 .env 파일을 찾아 그 안의 값을 프로세스 환경변수로 불러옴
key = os.environ["OPENAI_API_KEY"] # 현재 프로세스의 환경변수에서 키를 읽음. 변수가 없으면 KeyError 발생