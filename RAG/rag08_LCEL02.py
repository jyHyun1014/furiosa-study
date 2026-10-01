# LCEL = LangChain Expression Language

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

prompt = PromptTemplate.from_template("{topic}에 대해 {how} 설명해주세요")

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
model = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

chain = prompt | model

input = {"topic": "LCEL", "how": "초등학생도 이해하기 쉽고 간결하게"}

response = chain.invoke(input)
print(response)
print(response.content)
# LCEL은 **LangChain Expression Language**의 줄임말이에요.

# 쉽게 말하면, AI에게 일을 시킬 때 여러 단계를 **레고 블록처럼 연결하는 방법**이에요.

# 예를 들어:

# 1. 사용자의 질문을 받기  
# 2. AI에게 질문 보내기  
# 3. 답변을 보기 좋게 정리하기  

# 이 과정을 LCEL로 연결할 수 있어요.

# ```python
# 질문 | AI | 답변정리
# ```

# 여기서 `|` 기호는 “그다음 단계로 넘겨!”라는 뜻이에요.

# 즉, LCEL은 AI 프로그램의 작업 순서를 **간단하고 읽기 쉽게 이어 붙이는 문법**입니다.