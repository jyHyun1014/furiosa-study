# LangGraph의 핵심 개념 4가지
# Node    : 실제 작업을 수행하는 함수 또는 처리 단위
# Edge    : Node 간의 연결 관계 및 실행 흐름
# State   : 그래프의 Node들이 공유하는 데이터
# Reducer : State를 업데이트할 때 기존 값과 새로운 값을 병합하는 방법을 정의하는 함수

from typing import Annotated, TypedDict
from langgraph.graph import add_messages

# LangGraph에서 사용할 State 구조 정의
class MyState(TypedDict):
    # messages: 메시지 목록을 저장하는 State
    # Annotated: 타입에 추가적인 메타데이터(여기서는 Reducer)를 지정
    # list: 메시지들을 리스트 형태로 저장
    # add_messages: 기존 메시지와 새로운 메시지를 병합하는 Reducer
    #               메시지 ID가 다르면 추가하고, 같으면 기존 메시지를 새로운 메시지로 교체
    messages : Annotated[list, add_messages] # messages가 리스트 타입이며, LangGraph에서 해당 State 필드를 업데이트할 때 add_messages Reducer를 사용하도록 지정한다는 의미

from langchain_core.messages import AIMessage, HumanMessage

# HumanMessage: 사용자가 입력한 메시지
# AIMessage: AI가 생성한 메시지
# id: 각 메시지를 구분하기 위한 고유 식별자

# 1. 메시지 ID가 다른 경우: 기존 메시지에 새로운 메시지 추가
msg1 = [HumanMessage(content='안녕', id='1')] # 사용자 메시지 (ID: 1)
msg2 = [AIMessage(content='방가', id='2')] # AI 메시지 (ID: 2)


# add_messages(left, right)
# msg1을 기존 메시지 목록으로, msg2를 새로운 메시지 목록으로 전달
# 두 메시지의 ID가 다르므로 msg1 뒤에 msg2가 추가됨
results = add_messages(msg1, msg2)

print(results) 
# [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'), 
# AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2', tool_calls=[], invalid_tool_calls=[])]


############################################################################################

# 2. 메시지 ID가 같은 경우: 기존 메시지를 새로운 메시지로 교체
msg3 = [HumanMessage(content='안녕', id='3')] # 사용자 메시지 (ID: 3)
msg4 = [AIMessage(content='방가', id='3')] # AI 메시지 (ID: 3)

# msg3과 msg4의 ID가 동일하므로 기존 msg3이 msg4로 교체됨
# 따라서 결과에는 msg4만 남음
results2 = add_messages(msg3, msg4)

print(results2) 
# [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='3', tool_calls=[], invalid_tool_calls=[])]