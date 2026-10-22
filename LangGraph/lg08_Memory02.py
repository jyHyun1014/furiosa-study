# 8-1 카피

# LangGraph에 InMemorySaver를 적용하여 대화 기록을 메모리에 저장하고, 
# 동일한 thread_id를 통해 이전 대화 내용을 기억하며 Tool Calling을 수행하기

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


################# 1. Tool 정의 #################

# pip install langchain-tavily
from langchain_tavily import TavilySearch

# TavilySearch: 인터넷에서 최신 정보를 검색하는 외부 검색 도구
web_search = TavilySearch(
    max_results=3,
)

# Tool 단독 실행 테스트
result = web_search.invoke({
    "query": "삼성전자의 오늘 주가"
})

# print(result)
# {'query': '삼성전자의 오늘 주가', 
# 'follow_up_questions': None, 
# 'answer': None, 
# 'images': [], 
# 'results': [{'url': 'https://foxcg.com/entry/삼성전자-주가-전망-목표주가와-하락하는-이유-오늘-실시간-주가', 'title': '삼성전자 주가 전망, 목표주가와 하락하는 이유 - 오늘 실시간 주가 - 폭스씨지 FoxCG', 'content': '## 삼성전자 주가 흐름과 최근 하락 원인\n\n겜스고(GamsGo) 구독요금 할인 바로가기\n\n삼성전자는 최근 몇 년간 주가 하락의 큰 폭을 겪었습니다.  \n특히 2021년 이후 반도체 공급 과잉과 함께, 반도체 가격 하락 및 수요 감소가 이어지며 주가는 지속적으로 하락세를 보였습니다.\n\n2024년 기준으로 주가는 6만 원 선까지 떨어지며, 투자자들 사이에서는 \'5만전자\'라는 말까지 나올 정도로 큰 하락을 기록했습니다.\n\n외국인 투자자들의 지속적인 매도 압력도 삼성전자 주가 하락의 주요 원인 중 하나입니다.\n\n2023년부터 2024년까지 외국인은 수조 원 규모의 삼성전자 주식을 매도하며 주가 하방 압력을 가했습니다.\n\n특히 모건스탠리와 맥쿼리 등 외국계 증권사들이 삼성전자의 목표 주가를 하향 조정하면서 비관론이 확산되었습니다.\n\n 🔗 오늘 삼성전자 주가 바로가기 👉 \n\n"삼성전자 주가 전망, 목표주가와 하락하는 이유 - 오늘 실시간 주가"\n\n## 삼성전자 주가 하락의 구체적 원인 [...] 1. 적정 주가 평가: 현재 삼성전자의 주가가 과매도 상태에 있다는 분석이 많습니다. 2024년 10월 기준으로 삼성전자의 PBR(주가순자산비율)은 약 1.18배로, 이는 역사적으로 낮은 수준입니다. 전문가들은 현재 주가가 기업의 실제 가치 대비 과도하게 낮게 평가되고 있다는 의견을 제시하며, 매수 기회로 볼 수 있다고 분석하고 있습니다.\n2. 반도체 수출 증가와 실적 개선 가능성: 삼성전자의 2024년 3분기 반도체 수출 실적은 전년 동기 대비 41.4% 증가하며 사상 최대치를 기록했습니다. 특히 AI 서버와 데이터 센터의 신규 투자로 인해 메모리 반도체 수요가 꾸준히 증가하고 있으며, 이는 삼성전자의 실적 개선에 긍정적인 영향을 미칠 수 있습니다. [...] 1. 장기적 관점에서 매수 전략: 현재 주가 수준이 과매도 상태에 있다는 평가가 많으므로, 중장기적인 투자 관점에서 매수 기회로 볼 수 있습니다. 특히 반도체 수출 증가와 AI 기술의 발전은 삼성전자의 향후 성장을 뒷받침할 중요한 요소입니다.\n2. 리스크 관리: 외국인 매도세와 글로벌 경제 불확실성은 여전히 삼성전자 주가에 부정적인 영향을 줄 수 있습니다. 따라서 투자자들은 적절한 리스크 관리를 통해 포트폴리오를 다각화하는 것이 중요합니다.\n3. 반도체 시장 모니터링: 메모리 반도체와 HBM3E와 같은 고부가가치 제품의 수요와 공급을 지속적으로 모니터링하는 것이 필수적입니다. 반도체 시장의 변화는 삼성전자의 주가에 직접적인 영향을 미치므로, 관련 뉴스와 동향을 주의 깊게 살펴보는 것이 필요합니다.\n\n## 결론\n\n삼성전자의 주가는 최근 몇 년간 큰 폭의 하락을 겪었지만, 이는 반도체 시장의 하락 사이클과 외국인 투자자들의 매도 압력 등 복합적인 원인에 따른 것입니다.', 'score': 0.81875163, 'raw_content': None, 'id': 'a82bde-00'}, 
#               {'url': 'https://cconzirac.tistory.com/entry/%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80-%EC%98%A4%EB%8A%98-%EC%99%9C-%EC%98%AC%EB%9E%90%EC%9D%84%EA%B9%8C-%EC%83%81%EC%8A%B9-%EB%B0%B0%EA%B2%BD-3%EA%B0%80%EC%A7%80-%EC%A0%95%EB%A6%AC', 'title': '삼성전자 주가, 오늘 왜 올랐을까? 상승 배경 3가지 정리!', 'content': '본문 바로가기\n\n# 하루10분 주식공부\n\n카테고리 없음\n\n# 삼성전자 주가, 오늘 왜 올랐을까? 상승 배경 3가지 정리!\n\n by 상승씨 2025. 8. 7.\n\n## \n\n## \n\n## 📈 삼성전자 주가, 오늘 왜 올랐을까? 상승 배경 3가지 정리!\n\n오늘(8월 7일) 주식 시장을 보신 분들이라면  \n삼성전자 주가가 다시 상승세를 탄 것에 눈길이 갔을 거예요.\n\n"삼성전자 또 올라? 이유가 뭐지?"  \n저도 궁금해서 오늘 뉴스를 하나하나 챙겨봤는데요,  \n주요 원인을 크게 세 가지로 정리할 수 있었습니다.  \n지금부터 쉽게 풀어서 설명해드릴게요 😊\n\n### ✅ 1. 애플과의 협력 강화 소식\n\n가장 큰 뉴스는 바로 삼성전자가 미국 오스틴 공장에서 애플의 차세대 칩을 생산한다는 소식이었어요.\n\n이 칩은 애플의 신제품(아이폰 또는 아이패드)에 들어갈 것으로 보이며,  \n그만큼 삼성전자의 파운드리(반도체 위탁 생산) 기술력이 다시 주목받고 있다는 뜻이죠. [...] > ✔ 투자자들 입장에선 “삼성이 애플과 손잡았다면 믿고 투자할 만하지!”라는 생각이 들 수밖에요.\n\n이 뉴스가 공개되자마자 투자심리가 살아나면서  \n삼성전자뿐 아니라 반도체 관련주들도 함께 반등했답니다.\n\n### ✅ 2. 글로벌 불확실성 속에서도 긍정적인 흐름\n\n최근 미국에서는 반도체 관련 관세 이슈가 있었죠.  \n중국, 대만, 한국 등 주요 국가들에 영향을 줄 수 있는 뉴스였는데요,  \n놀랍게도 시장에서는 오히려 삼성전자의 미국 생산 확대가 리스크를 줄일 수 있다는 시각이 많았어요.\n\n즉, “삼성은 미국에서 칩도 만들고, 애플이랑도 잘 지내네? 괜찮겠는걸?” 하는 분위기인 거죠.\n\n> 📌 시장이 불안할수록 글로벌 신뢰도가 높은 기업이 주목받는 법이니까요.\n\n### ✅ 3. 외국인 투자자 매수 + 기대감 반영\n\n오늘 삼성전자 주가를 이끈 또 다른 요인은 바로  \n외국인 투자자들의 매수세였어요. [...] 삼성전자는 여전히 국내 주식 시장의 대표주자이고,  \n오늘처럼 글로벌 협력이나 기술 경쟁력이 부각되는 날에는  \n주가가 빠르게 반응하기도 해요.\n\n앞으로도 이런 뉴스 흐름을 잘 체크하면서,  \n투자 판단에 도움이 되셨으면 좋겠어요 😊\n\n저작자표시 비영리 변경금지  (새창열림)\n\n## 태그\n\n반도체, 반도체대장주, 삼성전자, 주린이공부, 주식공부\n\n---', 'score': 0.78532547, 'raw_content': None, 'id': '652cfc-01'}, 
#               {'url': 'https://foxcg.com/%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80-%EC%A0%84%EB%A7%9D-%EB%AA%A9%ED%91%9C%EC%A3%BC%EA%B0%80%EC%99%80-%ED%95%98%EB%9D%BD%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0-%EC%98%A4%EB%8A%98-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EC%A3%BC%EA%B0%80', 'title': '삼성전자주가전망｜주가 확인법·전망 체크 포인트 정리', 'content': 'A 아닙니다. 목표주가는 약속이나 보장이 아니며, 예상 실적이 달라지면 목표주가도 바뀝니다. 숫자 자체보다 산출 근거와 작성일을 확인하세요.\n\nQ 내일 삼성전자 주가를 알 수 있나요?\n\nA 없습니다. 누구도 다음 거래일의 주가를 정확히 예측할 수 없습니다. “내일 전망” 글은 예측이 아니라 확인 포인트로 읽는 것이 안전합니다.\n\nQ 실시간 주가는 어디서 확인하나요?\n\nA 네이버 금융 종목 페이지(종목코드 005930), 한국거래소(KRX), 증권사 MTS·HTS 앱, 삼성전자 IR 페이지에서 확인할 수 있습니다.\n\n📎 공식 출처: 삼성전자 IR · 한국거래소(KRX) · 에프앤가이드 (확인일 2026-10-17)\n\n겜스고 구독요금 할인 바로가기\n\n## 함께보면 좋은 글\n\n 2025년 공무원 봉급표 확정 및 9급, 8급, 7급 급여 총정리\n\n  ### 2025년 공무원 봉급표 확정 및 9급, 8급, 7급 급여 총정리 [...] | 확인 경로 | 확인할 수 있는 것 |\n --- |\n| 네이버 금융 종목 페이지 | 현재가·차트·거래량·뉴스·공시·증권사 리포트 목록 |\n| 한국거래소(KRX) | 시장 공식 시세·공시 정보 |\n| 증권사 MTS·HTS 앱 | 실시간 시세·호가·차트·리포트 (증권사 계좌 필요 여부는 앱별 상이) |\n| 삼성전자 IR 페이지 | 주가 정보·실적발표 자료·IR 일정 |\n\n삼성전자 종목코드는 005930입니다. 검색창에 종목코드나 종목명을 입력하면 종목 페이지로 바로 이동할 수 있습니다. 장중 시세와 장 마감 후 종가는 구분해서 확인하세요.\n\n🔗 네이버 금융 삼성전자 종목 페이지 · 한국거래소(KRX) · 삼성전자 IR\n\n## 삼성전자주가전망, 증권사 리포트·컨센서스 확인 방법 [...] 🔗 삼성전자 IR 실적발표 자료\n\n## 내일 주가·토론방 정보, 이렇게 걸러보기\n\n삼성전자주가전망 연관검색어에 “내일 삼성전자 주가 전망”이 있지만, 내일 주가를 정확히 예측할 수 있는 방법은 없습니다. 누구도 다음 거래일의 주가를 보장할 수 없으며, “내일 전망”을 내세우는 글은 예측이 아니라 확인 포인트로 읽어야 합니다.\n\n⚠️ 토론방·커뮤니티 정보 걸러보기\n\n① 출처 확인 — 익명 글보다 공식 공시·IR 자료·리포트 원문을 우선\n\n② “카더라” 주의 — 근거 없는 소문·미확인 유출 정보는 사실로 받아들이지 않기\n\n③ 수익 인증·유료방 유도 주의 — 수익을 과시하며 유료 가입을 유도하는 글은 경계\n\n④ 감정적 표현 경계 — “무조건”, “확실” 같은 단정적 표현이 많은 글일수록 검증 필요\n\n토론방은 시장 분위기를 파악하는 참고 자료일 뿐, 투자 판단의 근거가 될 수 없습니다. 중요한 결정 앞에서는 반드시 공식 자료와 리포트 원문으로 되돌아가 확인하세요.', 'score': 0.56719416, 'raw_content': None, 'id': 'b0b0a2-02'}], 
# 'response_time': 0.0, 
# 'request_id': 'fa4d9c62-6385-4b22-8f0b-4744f72cef6e'}

# 검색 결과 중 마지막 문서의 본문 출력
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tools =llm.bind_tools(tools)

################# 2. State 정의 #################

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 메모리 추가 첫번째
# InMemorySaver: 그래프의 State를 메모리(RAM)에 저장하는 Checkpointer
# 동일한 thread_id로 그래프를 실행하면 이전 대화 기록을 불러올 수 있음
# 프로그램이 종료되면 저장된 데이터는 사라짐
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]


################# 3. Node 정의 #################

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


################# 4. 그래프 구성, Node 추가 #################

# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))


################# 5. Edge 추가 #################

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", "chatbot")

################# 6. 그래프 컴파일 #################

# 메모리 추가 두 번째
# InMemorySaver 객체 생성: 그래프 실행 시 State를 저장하고 복원하는 역할
memory = InMemorySaver()

# checkpointer: 그래프의 State를 저장할 Checkpointer 지정
# 각 그래프 실행의 State를 저장하여 다음 실행에서 이전 대화 기록을 이어서 사용 가능
graph = graph_builder.compile(checkpointer=memory)

################# 7. 그래프 실행 #################

# 메모리 추가 세 번째
# thread_id: 대화 세션을 구분하는 식별자
# 동일한 thread_id를 사용하면 이전 대화 기록을 이어서 사용
# 서로 다른 thread_id를 사용하면 독립적인 대화 기록으로 관리
config = {"configurable": {"thread_id": "user01"}}


# 첫 번째 질문
# graph.invoke(): 그래프를 실행하고 최종 State 반환
# config에 thread_id를 전달하여 해당 대화 세션의 State를 저장
response = graph.invoke(
    {"messages": [("user", "내 이름은 김말티이야..")]},
    config=config,
)

# 최종 State의 messages에서 마지막 메시지(LLM 응답) 출력
print(response["messages"][-1].content)


print("===================== 두번째 질문 ===================")

# 두 번째 질문
# 첫 번째 질문과 동일한 thread_id를 사용하므로 이전 대화 기록을 자동으로 불러옴
# 기존 messages에 새로운 사용자 메시지가 추가된 상태로 그래프 실행
# 따라서 LLM은 첫 번째 질문에서 알려준 이름을 참고하여 답변할 수 있음
response = graph.invoke(
    {"messages": [("user", "내 이름이 뭐라고 했지?")]},
    config=config,
)

# 이전 대화 기록을 참고하여 생성한 LLM의 최종 응답 출력
print(response["messages"][-1].content)



# 반가워, 김말티! 😊  
# 이름이 정말 귀엽고 기억에 잘 남는다. 무엇을 도와줄까?
# ===================== 두번째 질문 ===================
# 김말티라고 했어! 😊