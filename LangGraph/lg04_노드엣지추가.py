# 3 카피

# LangGraph의 StateGraph를 이용하여 질문 접수(prepare), LLM 응답 생성(chatbot), 처리 완료(finish) 노드를 순차적으로 실행하고
# add_messages Reducer로 대화 메시지를 누적하는 AI 챗봇

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
    max_tokens=1000, # 최대 출력 토큰 수
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage



################# 1. State 정의 #################
# State: 그래프의 Node들이 공유하는 데이터
# TypedDict를 이용하여 State의 Key와 Value 타입 정의

class State(TypedDict):
    messages: Annotated[list, add_messages] # 대화 메시지 목록, add_messages Reducer 적용
    aaa: str # 추가 상태 데이터 (문자열)
    bbb: str # 추가 상태 데이터 (문자열)
    ccc: str # 추가 상태 데이터 (문자열)

# add_messages: 기존 메시지와 새로운 메시지를 병합하는 Reducer
# 메시지 ID가 다르면 추가하고, 같으면 기존 메시지를 새로운 메시지로 교체
# Reducer를 지정하지 않은 State 필드는 새로운 값으로 덮어씀


################# 2. node 정의 #################
# Node: 그래프에서 실제 작업을 수행하는 함수
# Node는 현재 State를 입력받고, 업데이트할 State 데이터를 딕셔너리로 반환

def prepare(state: State):
    return {"messages": [AIMessage(content="질문을 접수했습니다.")]}

def chatbot(state: State):
    # state["messages"]: 현재 State에 저장된 대화 메시지 목록
    # llm.invoke(): 메시지 목록을 LLM에 전달하여 AI 응답 생성
    # 반환된 AIMessage를 리스트에 담아 messages 필드 업데이트
    # add_messages Reducer가 기존 메시지와 새로운 AI 응답을 병합
    return {"messages": [llm.invoke(state["messages"])]}

def finish(state: State):
    return {"messages": [AIMessage(content="답변 처리가 완료되었습니다.")]}

################# 3. 그래프 구성 #################
# StateGraph: State를 기반으로 Node와 Edge를 구성하는 그래프 생성

graph_01 = StateGraph(State) # State 구조를 사용하는 그래프 생성

# add_node(노드명, 함수): 그래프에 Node 등록
graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)


################# 4. 엣지 추가 #################
# Edge: Node 간의 연결 관계 및 실행 순서를 정의
# START: 그래프 시작 지점
# END: 그래프 종료 지점

graph_01.add_edge(START, "prepare") # START → prepare
graph_01.add_edge("prepare", "chatbot") # prepare → chatbot
graph_01.add_edge("chatbot", "finish") # chatbot → finish
graph_01.add_edge("finish", END) # finish → END


################# 5. 그래프 컴파일 #################
# compile(): 정의한 Node와 Edge를 실행 가능한 그래프로 변환

graph = graph_01.compile()


################# 6. 그래프 실행 #################
# invoke(): 초기 State를 전달하여 그래프 실행
# START → prepare → chatbot → finish → END 순서로 실행

query = "삼성전자의 주가를 예측해줘"

# messages에 사용자 메시지를 전달하여 그래프 실행
# ("user", query)는 내부적으로 HumanMessage로 변환됨
response = graph.invoke({
    "messages": [("user", query)]
})
# graph.invoke()는 최종 State를 딕셔너리 형태로 반환
# add_messages Reducer에 의해 각 Node에서 생성한 메시지가 기존 목록에 추가됨

################# 7. 결과 출력 #################
for i, msg in enumerate(response['messages']):
    print(f"[{i}] {msg.type}: {msg.content}")

# [0] human: 삼성전자의 주가를 예측해줘
# [1] ai: 질문을 접수했습니다.
# [2] ai: 삼성전자 주가를 정확히 예측하는 것은 불가능하지만, **반도체 업황·HBM/AI 메모리 경쟁력·스마트폰 수요·환율·주주환원 정책**을 기준으로 방향성은 가늠할 수 있습니다.
# ...
# 결론적으로, **중장기적으로는 AI 메모리와 메모리 업황 회복이 삼성전자 주가의 핵심 상승 동력**입니다. 다만 HBM 경쟁력과 파운드리 개선이 기대보다 늦어지면 단기적으로는 주가가 기대에 못 미칠 수 있습니다.  
# 이는 투자 조언이 아니라 일반적 분석이며, 실제 투자 전에는 현재 주가, 실적 추정치, 본인의 투자 기간과 손실 감내 수준을 함께 고려해야 합니다.
# [3] ai: 답변 처리가 완료되었습니다.