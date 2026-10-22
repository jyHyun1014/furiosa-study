# LangGraph 병렬 실행 및 결과 병합 예제
#
# 실행 흐름:
#           ┌── search ────┐
# START ────┤              ├── merge ──── END
#           └── analysis ──┘
#
# Fan-out: 하나의 시작점에서 여러 Node로 분기하여 병렬 실행
# Fan-in: 여러 Node의 실행 결과를 하나의 Node로 모아 통합
#
# search와 analysis는 병렬 실행되고,
# 두 Node의 실행이 모두 완료되면 merge Node가 실행됨


from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State 정의
# State: 그래프의 Node들이 공유하는 데이터 구조
# 각 Node는 State에서 필요한 데이터를 읽고, 처리 결과를 State에 반환
class State(TypedDict):
    question: str         # 사용자가 입력한 질문
    search_result: str    # 검색 Node의 실행 결과
    analysis_result: str  # 분석 Node의 실행 결과
    final_result: str     # 검색 및 분석 결과를 통합한 최종 결과

# 2. 검색 Node 정의
# search_agent: 검색 작업을 수행하는 Node
# 실제 검색 대신 미리 정의된 문자열을 반환하는 예제
def search_agent(state: State):
    print("검색 에이전트 실행")
    # 검색 결과를 State의 search_result에 저장
    # Node는 변경할 State 필드만 딕셔너리로 반환하면 됨
    return {"search_result": "엔비디아는 AI GPU 시장의 주요 기업이다."}

# 3. 분석 Node 정의
# analysis_agent: 질문과 관련된 분석 작업을 수행하는 Node
# 실제 LLM 분석 대신 미리 정의된 문자열을 반환하는 예제
def analysis_agent(state: State):
    print("분석 에이전트 실행")
    # 분석 결과를 State의 analysis_result에 저장
    # search_agent와 서로 다른 State 필드를 수정하므로 병렬 실행 시 충돌하지 않음
    return {"analysis_result": "AI 반도체 수요 증가가 성장 요인이다."}

# 4. 결과 통합 Node 정의
# merge_agent: 검색 결과와 분석 결과를 하나로 통합하는 Node
# search와 analysis가 모두 완료된 후 실행
def merge_agent(state: State):
    # 병렬 실행된 두 Node의 결과를 State에서 가져와 문자열로 결합
    result = (
        state["search_result"] + "\n" + state["analysis_result"]
    )
    # 통합한 결과를 State의 final_result에 저장
    return {"final_result": result}

# 5. 그래프 구성
builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

# 6. Edge 추가
################# 병렬 실행 (Fan-out) #################

# START에서 search와 analysis Node로 동시에 분기
# 두 Node는 동일한 초기 State를 입력받아 병렬 실행
# 실행 순서는 보장되지 않음
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################# 결과 병합 (Fan-in) #################

# search와 analysis의 실행 결과를 merge Node로 전달
# 두 Node가 같은 실행 단계에서 병렬 수행되므로,
# 두 Node의 실행이 모두 완료된 후 merge Node가 한 번 실행됨
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)

# 7. 그래프 컴파일
graph = builder.compile()

# 8. 그래프 실행
# graph.invoke(): 그래프를 실행하고 최종 State 반환
# question에 사용자 질문을 전달
response = graph.invoke(
    {"question": "엔비디아의 미래 전망은?"}
)
print(response["final_result"])

# 분석 에이전트 실행
# 검색 에이전트 실행
# 엔비디아는 AI GPU 시장의 주요 기업이다.
# AI 반도체 수요 증가가 성장 요인이다.

# 참고:
# search와 analysis는 병렬 실행되므로 print() 출력 순서는 달라질 수 있음
# 하지만 merge에서는 두 Node의 결과가 모두 반영된 State를 사용

# 만약 두 Node가 동일한 State 필드를 동시에 수정한다면, 해당 필드에 적절한 Reducer를 정의하지 않은 경우 InvalidUpdateError가 발생할 수 있음

##############################
# 그래프 이미지 출력
##############################

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# graph.get_graph(): 컴파일된 LangGraph의 노드와 엣지 구조를 가져옴
# draw_mermaid_png(): 그래프 구조를 Mermaid 다이어그램으로 변환한 후 PNG 이미지로 렌더링
# 반환값은 PNG 이미지의 바이너리 데이터(bytes)
png_data = graph.get_graph().draw_mermaid_png()

# BytesIO(): PNG 바이너리 데이터를 메모리상의 파일 객체로 변환
# Image.open(): 메모리상의 PNG 데이터를 PIL 이미지 객체로 읽어옴
# 별도의 이미지 파일을 저장하지 않고 메모리에서 바로 처리
img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis("off")
plt.show()


