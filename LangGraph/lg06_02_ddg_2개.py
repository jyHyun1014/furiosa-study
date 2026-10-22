# 6-1 카피

# LangGraph에서 Tavily와 DuckDuckGo를 활용한 Tool Calling 예제

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

# pip install langchain-tavily
from langchain_tavily import TavilySearch

# TavilySearch: 인터넷에서 최신 정보를 검색하는 외부 검색 도구
web_search1 = TavilySearch(
    max_results=3,
    search_depth="basic",  # 검색 깊이 설정 (finance 검색에서 fast 사용 시 오류 발생 가능)
)

# pip install -U ddgs
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

# DuckDuckGoSearchRun: DuckDuckGo 검색 엔진을 이용하여 웹 검색을 수행하는 Tool
web_search2 = DuckDuckGoSearchRun()

# LLM에 연결할 도구 목록
tools = [web_search1, web_search2]

# bind_tools(): LLM에 Tool의 이름, 설명, 입력 스키마 등을 등록
# LLM이 질문을 분석하여 필요한 경우 도구 호출(tool_calls)을 생성할 수 있도록 설정
# 주의: bind_tools()는 Tool을 등록할 뿐, 실제 실행하지는 않음
llm_with_tools =llm.bind_tools(tools)


################# 2. State 정의 #################

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# State: 그래프의 Node들이 공유하는 데이터
class State(TypedDict):
    messages: Annotated[list, add_messages]


################# 3. Node 정의 #################

# chatbot Node: 사용자 질문과 대화 내역을 LLM에 전달
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


################# 4. 그래프 구성, Node 추가 #################

# StateGraph: State를 기반으로 Node와 Edge를 구성하는 그래프 생성
graph_builder = StateGraph(State)

# chatbot Node 등록
graph_builder.add_node("chatbot", chatbot)

# ToolNode: LLM이 요청한 도구를 실제로 실행하는 Node
# AIMessage의 response.tool_calls를 읽고 해당 도구를 실행
# 실행 결과를 ToolMessage로 반환하여 State의 messages에 추가
graph_builder.add_node("tools", ToolNode(tools))


################# 5. Edge 추가 #################

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
# response = graph.invoke({
#     "messages": [("user", query)]
# })


# graph.stream(): 그래프를 실행하면서 각 Node의 실행 결과를 순차적으로 반환
# stream_mode="updates": 각 Node가 State에 반영하는 변경 사항만 반환
for event in graph.stream(
    {"messages": [("user", query)]},
    stream_mode="updates"
):
    for node_name, node_output in event.items():

        # 실행된 Node 이름 출력
        print(f"\n===== {node_name} Node 실행 =====")

        for message in node_output.get("messages", []):

            # LLM이 요청한 Tool 확인
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    print(f"Tool 호출 요청: {tool_call['name']}")
                    print(f"Tool 입력 인자: {tool_call['args']}")

            # ToolNode에서 실제 실행된 Tool 확인
            elif message.type == "tool":
                print(f"실행된 Tool: {message.name}")
                print(f"Tool 실행 결과: {str(message.content)[:500]}")

            # LLM이 생성한 일반 응답
            elif message.type == "ai":
                print(f"AI 응답: {message.content}")


# ===== chatbot Node 실행 =====
# Tool 호출 요청: tavily_search
# Tool 입력 인자: {'query': '삼성전자 005930 오늘 주가 2026년 3월 31일', 'topic': 'finance', 'search_depth': 'basic'}

# ===== tools Node 실행 =====
# 실행된 Tool: tavily_search
# Tool 실행 결과: {"query": "삼성전자 005930 오늘 주가 2026년 3월 31일", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.instagram.com/popular/%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EA%B8%B0%EC%88%A0%EC%A0%81-%EB%B6%84%EC%84%9D-%EC%A7%80%EC%A7%80%EC%84%A0-%EC%A0%80%ED%95%AD%EC%84%A0-2026%EB%85%84-3%EC%9B%94", "title": "삼성전자 기술적 분석 지지선 저항선 2026년 3월", "content": "# 삼성전자 기술적 분석 지지선 저항선 2026년 3월. ⚠️ ✅ 체크 포인트 1️⃣ 낮아지는 고점: 22만 원 → 21만 원 (추세 약화) 2️⃣ 이평선 이탈: 20일 이동평균선의 가파른 하락 전환 3️⃣ 지표 괴리:

# ===== chatbot Node 실행 =====
# Tool 호출 요청: duckduckgo_search
# Tool 입력 인자: {'query': 'site:finance.naver.com/item/main.nhn?code=005930 삼성전자 현재가'}

# ===== tools Node 실행 =====
# 실행된 Tool: duckduckgo_search
# Tool 실행 결과: 

# ===== chatbot Node 실행 =====
# AI 응답: 삼성전자(005930)는 검색된 최신 장마감 정보 기준으로 **186,200원**입니다.  
# 전 거래일 대비 **+4.37% 상승**한 가격으로, **2026년 4월 3일 장 마감 기준**입니다.

# 실시간 장중 가격은 변동될 수 있으니 [네이버페이 증권 삼성전자](https://finance.naver.com/item/main.nhn?code=005930)에서 확인하세요.