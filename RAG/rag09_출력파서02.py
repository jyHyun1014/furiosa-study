# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

# 상황:
{question}

# FORMAT:
- 영어회화 :
- 한글번역 :
"""

prompt = PromptTemplate.from_template(template)

model = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {"question": "저는 부산에서 물밀면을 먹고 싶어요"}

response = chain.invoke(input)
print(response)
# - 영어회화 :  
# A: Excuse me, where can I eat good mul milmyeon in Busan?  
# B: You can try the restaurant near Seomyeon Station. It’s famous for its cold mul milmyeon.  
# A: Great! I’d like one bowl of mul milmyeon, please.  
# B: Would you like it spicy?  
# A: A little spicy, please.  

# - 한글번역 :  
# A: 실례합니다, 부산에서 맛있는 물밀면을 어디서 먹을 수 있나요?  
# B: 서면역 근처의 식당에 가 보세요. 그곳은 시원한 물밀면으로 유명해요.  
# A: 좋아요! 물밀면 한 그릇 주세요.  
# B: 맵게 드릴까요?  
# A: 조금 맵게 해 주세요.