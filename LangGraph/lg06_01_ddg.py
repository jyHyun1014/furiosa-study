# LangGraph에서 DuckDuckGo 웹 검색 Tool을 활용하는 Tool Calling 예제

################# 0. LLM 설정 #################

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0, # 응답의 무작위성 설정 (0에 가까울수록 일관된 응답)
    max_tokens=2000, # 최대 출력 토큰 수
    api_key=api_key,
    base_url=base_url,
)


################# 1. Tool 정의 #################

# pip install -U ddgs
from langchain_community.tools import DuckDuckGoSearchRun

# DuckDuckGoSearchRun: DuckDuckGo 검색 엔진을 이용하여 웹 검색을 수행하는 Tool
# 검색어를 입력받아 관련 웹페이지의 검색 결과를 문자열로 반환
# API Key 없이 사용 가능
web_search = DuckDuckGoSearchRun()

# Tool 단독 실행 테스트
result = web_search.invoke({
    "query": "삼성전자의 오늘 주가"
})
print(result)
# 삼성전자의 주가, 005930 주식, 차트, 기술적 분석, 실적 자료 등 223,500로 거래를 마감한 삼성전자 시세에 대한 자세한 정보를 확인해 보세요. 오늘의 시세뿐만 아니라 뉴스와 분석, 재정 상황, 포럼 등도 확인해 보세요. Google Finance에서 삼성전자의 실시간 주식 시세, 뉴스, 재무 세부정보, 분석을 확인하세요. 실시간 삼성전자 (005930) 주가를 광고 없는 깔끔한 화면에서 확인해보세요! 현재 주가는 224,500원입니다. 실시간 차트, 최신 뉴스, AI 분석 및 더 많은 정보를 알파스퀘어에서 확인해보세요. 삼성전자 급등·신고가·중요 공시 알림 받기 텔레그램 · 무료 최근 공시 공시 레이더 → 10.16★★ 주식등의대량보유상황보고서 (일반)10.08★★ 주식등의대량보유상황보고서 (일반)10.08★★ 연결재무제표기준 영업실적 잠정치 발표 일봉 주봉 월봉 1달 3달 1년 RSI ... Samsung Electronics Co., Ltd (005930/KRW)의 오늘 주가는 211,500 KRW이며, 지난 24시간 변동률은 -2.98%입니다. Samsung Electronics Co., Ltd 주식의 실시간 차트, 일중 흐름, 주요 뉴스 및 핵심 펀더멘털 지표를 확인해 보세요.

# exit() # Tool 결과만 확인하고 아래 그래프는 실행하지 않을 때


################# 2. LLM에 Tool 연결 #################

# LLM에 연결할 도구 목록
tools = [web_search]

# bind_tools(): LLM에 Tool의 이름, 설명, 입력 스키마 등을 등록
# LLM이 질문을 분석하여 필요한 경우 도구 호출(tool_calls)을 생성할 수 있도록 설정
# 주의: bind_tools()는 Tool을 등록할 뿐, 실제 실행하지는 않음
llm_with_tools =llm.bind_tools(tools)

################# 3. State 정의 #################

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# State: 그래프의 Node들이 공유하는 데이터
class State(TypedDict):
    messages: Annotated[list, add_messages]


################# 4. Node 정의 #################

# chatbot Node: 사용자 질문과 대화 내역을 LLM에 전달
def chatbot(state: State):
    # llm_with_tools.invoke(): 도구 사용이 가능한 LLM 호출
    # LLM이 도구가 필요하다고 판단하면 tool_calls가 포함된 AIMessage 반환
    # 검색이 필요하지 않으면 일반적인 AIMessage 반환
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 1. Tool이 필요 없는 경우
# - LLM이 직접 답변 생성
# - response.content에 답변 저장
# - response.tool_calls는 빈 리스트 []
# 2. Tool이 필요한 경우
# - LLM이 도구 호출 요청 생성
# - response.tool_calls에 도구 이름과 인자 저장
# - 실제 도구 실행은 별도로 필요 (ToolNode 등)


################# 5. 그래프 구성, Node 추가 #################

# StateGraph: State를 기반으로 Node와 Edge를 구성하는 그래프 생성
graph_builder = StateGraph(State)

# chatbot Node 등록
graph_builder.add_node("chatbot", chatbot)

# ToolNode: LLM이 요청한 도구를 실제로 실행하는 Node
# AIMessage의 response.tool_calls를 읽고 해당 도구를 실행
# 실행 결과를 ToolMessage로 반환하여 State의 messages에 추가
graph_builder.add_node("tools", ToolNode(tools))


################# 6. Edge 추가 #################

# START → chatbot
# 그래프 실행 시 chatbot Node부터 시작
graph_builder.add_edge(START, "chatbot")


# 조건에 따라 다음에 실행할 Node를 결정
# 중요: 도구 사용 여부는 LLM이 판단하고, tools_condition은 LLM이 생성한 response.tool_calls의 존재 여부를 확인하여 분기
graph_builder.add_conditional_edges(
    "chatbot", # 조건을 검사할 Node
    tools_condition # 마지막 AIMessage에 tool_calls가 있는지 확인하고 다음 Node를 결정하는 조건 함수
    # tool_calls가 있으면 "tools" 반환 → ToolNode 실행
    # tool_calls가 없으면 END 반환 → 그래프 종료
)
# tools_condition은 기본적으로 "tools" 또는 END를 반환
# 따라서 END로 연결하는 Edge를 별도로 추가할 필요 없음


# tools → chatbot
# ToolNode에서 도구 실행이 완료되면 chatbot Node를 다시 실행
# 도구 실행 결과(ToolMessage)를 LLM에 전달하여 최종 답변 생성
# LLM이 추가 도구 호출을 요청하면 다시 tools로 이동할 수도 있음
graph_builder.add_edge("tools", "chatbot")

################# 6. 그래프 컴파일 #################

# 정의한 Node와 Edge를 실행 가능한 그래프로 변환
graph = graph_builder.compile()

################# 7. 그래프 실행 #################

query = "삼성전자의 오늘 주가를 알려줘"

# 초기 State에 사용자 질문을 전달하여 그래프 실행
# ("user", query)는 내부적으로 HumanMessage로 변환
response = graph.invoke({
    "messages": [("user", query)]
})

# 최종 State의 마지막 AIMessage 출력
# Tool Calling이 수행되었다면 도구 실행 결과를 참고하여 생성한 답변
print("AI :", response["messages"][-1].content)

# AI : 삼성전자(005930) 현재 주가는 **약 224,500원**으로 확인됩니다.
# 다만 국내 주가는 장중 변동하며, 조회처에 따라 시세 지연(보통 20분)이 있을 수 있습니다. 정확한 실시간 가격은 KRX·증권사 앱·네이버금융 등에서 확인하세요.