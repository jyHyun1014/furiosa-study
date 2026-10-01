# LCEL = LangChain Expression Language

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요")

# LangChain의 OpenAI 연동은 API 키를 명시적으로 받지 않으면 기본적으로 OPENAI_API_KEY 환경 변수를 확인함
model = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

chain = prompt | model

input = {"topic": "양자컴퓨터 학습 원리"}

response = chain.invoke(input)
print(response)
# content='양자컴퓨터는 **아주 작은 입자(전자, 빛 등)가 따르는 양자역학 법칙**을 계산에 이용하는 컴퓨터입니다. 핵심 원리는 크게 4가지로 이해하면 쉽습니다.\n\n## 1. 비트 대신 큐비트(Qubit)를 씁니다\n\n일반 컴퓨터의 정보 단위는 **비트(bit)**입니다.\n\n- 비트: `0` 또는 `1`\n- 큐비트: 측정하기 전에는 `0`과 `1`의 가능성을 함께 가질 수 있음\n\n이를 흔히 동전으로 비유합니다.\n\n- 일반 비트: 책상 위에 놓인 동전 → 앞면 또는 뒷면\n- 큐비트: 공중에서 빙글빙글 도는 동전 → 앞면과 뒷면이 모두 가능해 보이는 상태\n\n이 상태를 **중첩(superposition)**이라고 합니다.\n\n단, “0과 1을 동시에 확정적으로 계산한다”는 뜻은 아닙니다.  \n측정하면 결국 `0` 또는 `1` 하나만 나오며, 대신 계산 과정에서 각 결과가 나올 **확률과 파동의 성질**을 조절할 수 있습니다.\n\n---\n\n## 2. 여러 큐비트는 서로 연결될 수 있습니다: 얽힘\n\n두 개 이상의 큐비트가 특별하게 연결되면 **얽힘(entanglement)** 상태가 됩니다.\n\n예를 들어 두 큐비트가 얽혀 있으면, 각각을 따로 보는 것보다 **둘의 관계 전체**가 중요해집니다.\n\n일반 비트 2개는 다음 네 가지 상태 중 하나입니다.\n\n- `00`\n- `01`\n- `10`\n- `11`\n\n하지만 얽힌 큐비트는 예를 들어 다음처럼 동작할 수 있습니다.\n\n- 측정하면 `00` 또는 `11`만 나오도록 연결됨\n- 한쪽 결과를 보면 다른 쪽 결과와 강하게 연관됨\n\n이 성질 덕분에 양자컴퓨터는 많은 변수 사이의 복잡한 관계를 표현하는 데 잠재력이 있습니다.\n\n---\n\n## 3. 정답의 확률을 키우는 핵심: 간섭\n\n양자컴퓨터가 빠를 수 있는 가장 중요한 이유는 **간섭(interference)**입니다.\n\n파도 두 개가 만나면:\n\n- 높이와 높이가 만나면 더 커짐 → **보강 간섭**\n- 높이와 낮음이 만나면 상쇄됨 → **상쇄 간섭**\n\n양자컴퓨터도 비슷합니다. 알고리즘을 설계해서\n\n- 정답에 해당하는 경우는 확률이 커지게 하고\n- 오답에 해당하는 경우는 서로 상쇄되게 합니다.\n\n즉, 양자컴퓨터는 단순히 “모든 답을 동시에 계산해서 정답을 바로 찾는 기계”라기보다,  \n**정답이 나올 가능성을 높이도록 양자 상태를 조작하는 기계**라고 보는 편이 정확합니다.\n\n---\n\n## 4. 마지막에는 측정해서 일반적인 답을 얻습니다\n\n계산이 끝나면 큐비트를 **측정(measurement)**합니다.\n\n측정하는 순간 큐비트의 중첩 상태는 하나의 결과로 정해집니다.\n\n예를 들어 어떤 계산 후:\n\n- `00`이 나올 확률 5%\n- `01`이 나올 확률 10%\n- `10`이 나올 확률 80%\n- `11`이 나올 확률 5%\n\n이라면 여러 번 실행했을 때 `10`이 가장 자주 나옵니다.  \n그래서 실제 양자 알고리즘은 보통 여러 번 실행해 결과를 통계적으로 확인합니다.\n\n---\n\n# 전체 과정을 아주 간단히 보면\n\n1. 큐비트를 준비한다  \n2. 중첩 상태를 만든다  \n3. 얽힘과 양자 게이트를 이용해 상태를 변화시킨다  \n4. 간섭을 이용해 정답 확률을 높인다  \n5. 측정해서 결과를 얻는다  \n\n여기서 **양자 게이트**는 일반 컴퓨터의 AND, OR 같은 논리 회로에 해당합니다. 다만 양자 게이트는 큐비트의 확률 상태와 위상을 바꾸는 방식으로 작동합니다.\n\n---\n\n# 예: 미로 찾기 비유\n\n일반 컴퓨터는 미로의 길을 하나씩 검사하는 방식으로 생각할 수 있습니다.\n\n양자컴퓨터는 여러 길의 가능성을 양자 상태로 표현한 뒤,\n\n- 막다른 길은 서로 상쇄되도록 만들고\n- 정답 경로는 강화되도록 만들어\n\n마지막 측정에서 정답 경로가 나올 확률을 높이는 방식입니다.\n\n다만 모든 미로 문제에서 무조건 빠른 것은 아닙니다.  \n특정 수학 문제, 최적화 문제, 화학·물질 시뮬레이션, 암호 관련 문제 등에서 특히 강점이 기대됩니다.\n\n---\n\n# 중요한 오해 3가지\n\n### 1. 양자컴퓨터가 모든 문제를 빠르게 푸는 것은 아닙니다\n어떤 문제는 일반 컴퓨터가 훨씬 효율적입니다.\n\n### 2. 큐비트는 불안정합니다\n외부 열, 진동, 전자기파 등에 매우 민감해 오류가 생기기 쉽습니다. 이를 **양자 오류 보정**으로 해결하려는 연구가 중요합니다.\n\n### 3. 아직 일반 컴퓨터를 완전히 대체하지는 못합니다\n현재는 연구·실험 단계와 일부 특수 활용 단계에 가깝습니다. 앞으로는 일반 컴퓨터와 양자컴퓨터가 역할을 나누어 함께 쓰일 가능성이 큽니다.\n\n---\n\n한 문장으로 요약하면:\n\n> 양자컴퓨터는 중첩, 얽힘, 간섭이라는 양자 현상을 이용해 정답의 확률을 높이도록 계산하는 컴퓨터입니다.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 1372, 'prompt_tokens': 18, 'total_tokens': 1390, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 63, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETgQCEWsxcheCuxEFP4GNYXavR3p6', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0f09e-19bd-7fe0-b7f5-eefc84fae8f2-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 18, 'output_tokens': 1372, 'total_tokens': 1390, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 63}}
print(response.content)