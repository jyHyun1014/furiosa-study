# LangGraph를 이용한 기본 챗봇 구현
# StateGraph로 그래프를 생성하고, 사용자 메시지를 LLM에 전달하여 응답을 생성하는 예제
# 실행 흐름: START → chatbot(Node) → END

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

# State 데이터 구조 예시
# state = {
#     "messages": [],
#     "aaa": "abc",
#     "bbb": "bcd",
#     "ccc": "cde",
# }


################# 2. node 정의 #################
# Node: 그래프에서 실제 작업을 수행하는 함수
# Node는 현재 State를 입력받고, 업데이트할 State 데이터를 딕셔너리로 반환

def chatbot(state: State):
    # state["messages"]: 현재 State에 저장된 대화 메시지 목록
    # llm.invoke(): 메시지 목록을 LLM에 전달하여 AI 응답 생성
    # 반환된 AIMessage를 리스트에 담아 messages 필드 업데이트
    # add_messages Reducer가 기존 메시지와 새로운 AI 응답을 병합
    return {"messages": [llm.invoke(state["messages"])]}


################# 3. 그래프 구성 #################
# StateGraph: State를 기반으로 Node와 Edge를 구성하는 그래프 생성

graph_01 = StateGraph(State) # State 구조를 사용하는 그래프 생성

# add_node(노드명, 함수): 그래프에 Node 등록
graph_01.add_node("chatbot", chatbot)


################# 4. 엣지 추가 #################
# Edge: Node 간의 연결 관계 및 실행 순서를 정의
# START: 그래프 시작 지점
# END: 그래프 종료 지점

graph_01.add_edge(START, "chatbot") # START → chatbot
graph_01.add_edge("chatbot", END) # chatbot → END


################# 5. 그래프 컴파일 #################
# compile(): 정의한 Node와 Edge를 실행 가능한 그래프로 변환

graph = graph_01.compile()


################# 6. 그래프 실행 #################
# invoke(): 초기 State를 전달하여 그래프 실행
# START → chatbot → END 순서로 실행

query = "삼성전자의 주가를 예측해줘"

# messages에 사용자 메시지를 전달하여 그래프 실행
# ("user", query)는 내부적으로 HumanMessage로 변환됨
response = graph.invoke({
    "messages": [("user", query)]
})
# graph.invoke()는 최종 State를 딕셔너리 형태로 반환
# messages에는 사용자 메시지와 AI 응답이 저장됨

print(response) # 최종 State 전체 출력
# {'messages': [HumanMessage(content='삼성전자의 주가를 예측해줘', additional_kwargs={}, response_metadata={}, id='8ba41cc3-3410-4ae9-b400-6fc6225112e2'), AIMessage(content='삼성전자(005930)의 주가를 **정확히 예측하는 것은 불가능**합니다. 특히 반도체 업황, 글로벌 금리, 환율, 미·중 규제, AI 메모리(HBM) 경쟁력 같은 변수가 단기간에 크게 영향을 줍니다. 아래는 투자 판단을 위한 **시나리오 기반 전망 틀**입니다.\n\n### 주가 상승 요인\n- **HBM·DDR5 등 AI 메모리 매출 확대**: 엔비디아 등 AI 서버 생태계에서 HBM 공급 경쟁력이 개선될 경우 핵심 호재입니다.\n- **메모리 반도체 가격 상승**: DRAM·NAND 가격이 오르고 감산 효과가 지속되면 실적 레버리지가 큽니다.\n- **파운드리 수율 및 대형 고객 확보**: 첨단 공정 수율 개선과 외부 고객 확대는 기업가치 재평가 요인입니다.\n- **주주환원 강화**: 자사주 매입·소각, 배당 확대 등은 하방을 지지할 수 있습니다.\n- **원/달러 환율 상승**: 수출 비중이 큰 만큼 일정 범위의 원화 약세는 이익에 우호적일 수 있습니다.\n\n### 주가 하락 요인\n- **HBM 인증·공급 지연 또는 경쟁사 우위 지속**\n- **메모리 가격 재하락 및 공급 과잉**\n- **스마트폰·가전 수요 부진**\n- **파운드리 적자 확대와 설비투자 부담**\n- **미국 금리 상승, 글로벌 경기침체, 미·중 갈등 심화**\n\n### 기간별 관점\n- **단기(수주~3개월)**: 외국인 수급, 미국 반도체주 흐름, 환율, 메모리 가격 뉴스에 따라 변동성이 클 가능성이 높습니다.\n- **중기(6~12개월)**: HBM 공급 확대와 메모리 업황의 실제 실적 반영 여부가 핵심입니다.\n- **장기(2~3년 이상)**: AI 메모리 점유율, 파운드리 경쟁력, 주주환원 정책이 주가의 구조적 방향을 결정할 가능성이 큽니다.\n\n### 확인하면 좋은 지표\n1. 분기별 **반도체(DS) 영업이익**\n2. HBM 고객사 인증·출하 관련 공시 및 실적발표 내용\n3. DRAM/NAND 현물·계약 가격 추세\n4. 외국인 순매수와 원/달러 환율\n5. 경쟁사(SK하이닉스, TSMC, 마이크론) 실적과 가이던스\n\n결론적으로, 삼성전자는 AI 메모리 및 메모리 업황 회복이 이어질 경우 중장기 상승 여력이 있을 수 있지만, 단기적으로는 HBM 경쟁력과 반도체 가격 흐름에 따라 큰 변동이 가능한 종목입니다.  \n원하시면 **현재 주가를 기준으로 기술적 지지·저항 구간**, 또는 **보수적·기준·낙관 시나리오별 목표 범위 산정 방법**을 정리해드릴 수 있습니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 785, 'prompt_tokens': 17, 'total_tokens': 802, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 63, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-EacWoRpPGUaSZuceOQ02y6hc6wlkx', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a15329-ebdd-75c0-b8a4-2d01fd7ee355-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 17, 'output_tokens': 785, 'total_tokens': 802, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 63}})]}

print(response["messages"][0]) # 첫 번째 메시지: HumanMessage (사용자 질문)
# content='삼성전자의 주가를 예측해줘' additional_kwargs={} response_metadata={} id='8ba41cc3-3410-4ae9-b400-6fc6225112e2'

print(response["messages"][1]) # 두 번째 메시지: AIMessage (AI 응답)
# content='삼성전자(005930)의 주가를 **정확히 예측하는 것은 불가능**합니다. 특히 반도체 업황, 글로벌 금리, 환율, 미·중 규제, AI 메모리(HBM) 경쟁력 같은 변수가 단기간에 크게 영향을 줍니다. 아래는 투자 판단을 위한 **시나리오 기반 전망 틀**입니다.\n\n### 주가 상승 요인\n- **HBM·DDR5 등 AI 메모리 매출 확대**: 엔비디아 등 AI 서버 생태계에서 HBM 공급 경쟁력이 개선될 경우 핵심 호재입니다.\n- **메모리 반도체 가격 상승**: DRAM·NAND 가격이 오르고 감산 효과가 지속되면 실적 레버리지가 큽니다.\n- **파운드리 수율 및 대형 고객 확보**: 첨단 공정 수율 개선과 외부 고객 확대는 기업가치 재평가 요인입니다.\n- **주주환원 강화**: 자사주 매입·소각, 배당 확대 등은 하방을 지지할 수 있습니다.\n- **원/달러 환율 상승**: 수출 비중이 큰 만큼 일정 범위의 원화 약세는 이익에 우호적일 수 있습니다.\n\n### 주가 하락 요인\n- **HBM 인증·공급 지연 또는 경쟁사 우위 지속**\n- **메모리 가격 재하락 및 공급 과잉**\n- **스마트폰·가전 수요 부진**\n- **파운드리 적자 확대와 설비투자 부담**\n- **미국 금리 상승, 글로벌 경기침체, 미·중 갈등 심화**\n\n### 기간별 관점\n- **단기(수주~3개월)**: 외국인 수급, 미국 반도체주 흐름, 환율, 메모리 가격 뉴스에 따라 변동성이 클 가능성이 높습니다.\n- **중기(6~12개월)**: HBM 공급 확대와 메모리 업황의 실제 실적 반영 여부가 핵심입니다.\n- **장기(2~3년 이상)**: AI 메모리 점유율, 파운드리 경쟁력, 주주환원 정책이 주가의 구조적 방향을 결정할 가능성이 큽니다.\n\n### 확인하면 좋은 지표\n1. 분기별 **반도체(DS) 영업이익**\n2. HBM 고객사 인증·출하 관련 공시 및 실적발표 내용\n3. DRAM/NAND 현물·계약 가격 추세\n4. 외국인 순매수와 원/달러 환율\n5. 경쟁사(SK하이닉스, TSMC, 마이크론) 실적과 가이던스\n\n결론적으로, 삼성전자는 AI 메모리 및 메모리 업황 회복이 이어질 경우 중장기 상승 여력이 있을 수 있지만, 단기적으로는 HBM 경쟁력과 반도체 가격 흐름에 따라 큰 변동이 가능한 종목입니다.  \n원하시면 **현재 주가를 기준으로 기술적 지지·저항 구간**, 또는 **보수적·기준·낙관 시나리오별 목표 범위 산정 방법**을 정리해드릴 수 있습니다.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 785, 'prompt_tokens': 17, 'total_tokens': 802, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 63, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-EacWoRpPGUaSZuceOQ02y6hc6wlkx', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a15329-ebdd-75c0-b8a4-2d01fd7ee355-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 17, 'output_tokens': 785, 'total_tokens': 802, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 63}}

print(response["messages"][1].content) # AI 응답의 텍스트 내용
# 삼성전자(005930)의 주가를 **정확히 예측하는 것은 불가능**합니다. 특히 반도체 업황, 글로벌 금리, 환율, 미·중 규제, AI 메모리(HBM) 경쟁력 같은 변수가 단기간에 크게 영향을 줍니다. 아래는 투자 판단을 위한 **시나리오 기반 전망 틀**입니다.

# ### 주가 상승 요인
# - **HBM·DDR5 등 AI 메모리 매출 확대**: 엔비디아 등 AI 서버 생태계에서 HBM 공급 경쟁력이 개선될 경우 핵심 호재입니다.
# - **메모리 반도체 가격 상승**: DRAM·NAND 가격이 오르고 감산 효과가 지속되면 실적 레버리지가 큽니다.
# - **파운드리 수율 및 대형 고객 확보**: 첨단 공정 수율 개선과 외부 고객 확대는 기업가치 재평가 요인입니다.
# - **주주환원 강화**: 자사주 매입·소각, 배당 확대 등은 하방을 지지할 수 있습니다.
# - **원/달러 환율 상승**: 수출 비중이 큰 만큼 일정 범위의 원화 약세는 이익에 우호적일 수 있습니다.

# ### 주가 하락 요인
# - **HBM 인증·공급 지연 또는 경쟁사 우위 지속**
# - **메모리 가격 재하락 및 공급 과잉**
# - **스마트폰·가전 수요 부진**
# - **파운드리 적자 확대와 설비투자 부담**
# - **미국 금리 상승, 글로벌 경기침체, 미·중 갈등 심화**

# ### 기간별 관점
# - **단기(수주~3개월)**: 외국인 수급, 미국 반도체주 흐름, 환율, 메모리 가격 뉴스에 따라 변동성이 클 가능성이 높습니다.
# - **중기(6~12개월)**: HBM 공급 확대와 메모리 업황의 실제 실적 반영 여부가 핵심입니다.
# - **장기(2~3년 이상)**: AI 메모리 점유율, 파운드리 경쟁력, 주주환원 정책이 주가의 구조적 방향을 결정할 가능성이 큽니다.

# ### 확인하면 좋은 지표
# 1. 분기별 **반도체(DS) 영업이익**
# 2. HBM 고객사 인증·출하 관련 공시 및 실적발표 내용
# 3. DRAM/NAND 현물·계약 가격 추세
# 4. 외국인 순매수와 원/달러 환율
# 5. 경쟁사(SK하이닉스, TSMC, 마이크론) 실적과 가이던스

# 결론적으로, 삼성전자는 AI 메모리 및 메모리 업황 회복이 이어질 경우 중장기 상승 여력이 있을 수 있지만, 단기적으로는 HBM 경쟁력과 반도체 가격 흐름에 따라 큰 변동이 가능한 종목입니다.  
# 원하시면 **현재 주가를 기준으로 기술적 지지·저항 구간**, 또는 **보수적·기준·낙관 시나리오별 목표 범위 산정 방법**을 정리해드릴 수 있습니다.

print(response["messages"][0].type) # human

print(response["messages"][1].type) # ai