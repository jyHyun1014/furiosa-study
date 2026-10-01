# 윈도우에서 시스템 환경변수 설정하고 실행되는지 확인하기

from langchain_openai import ChatOpenAI
import os

# 설정 → 시스템 환경변수 편집 → Admin에 사용자 변수 → 새로 만들기
# 변수 이름: OPENAI_API_KEY
# 변수 값: 발급받은API키

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    # openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?
