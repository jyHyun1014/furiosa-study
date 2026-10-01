from langchain_openai import ChatOpenAI
import os

# 환경변수 설정. 파이썬 프로세스가 실행되는 동안 사용 가능
os.environ["OPENAI_API_KEY"] = "발급받은API키"

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    # openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?
