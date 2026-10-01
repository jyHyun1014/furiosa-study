# .env 파일에서 설정한 API KEY 값 불러오기

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
    # openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?


