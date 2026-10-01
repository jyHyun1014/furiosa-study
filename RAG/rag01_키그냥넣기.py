from langchain_openai import ChatOpenAI

openai_api_key = "발급받은API키"

llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
# print(response) # content='안녕하세요! 무엇을 도와드릴까요?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLnzsLPxMjrJbRBndNYUhf8RBn2', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-f535-7c11-9133-0c1f74c42a00-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?

response = llm.invoke("안녕하세요. 나는 윤영선이야. 이해했니?")
print(response.content) # 안녕하세요, 윤영선님. 네, 이해했습니다.

response = llm.invoke("나는 누구게?")
print(response.content) # 아직은 단서가 없어서 모르겠어요 😄 힌트를 하나만 주세요!
