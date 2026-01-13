from pydantic_ai.models.openai import OpenAIModel
from zhipu_model import ZhipuModel
from dotenv import load_dotenv
import os

load_dotenv()

def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')
    provider = os.getenv("PROVIDER", 'openai')
    
    if provider == 'zhipu':
        return ZhipuModel(
            llm,
            base_url=base_url,
            api_key=api_key
        )
    elif provider == 'openai':
        return OpenAIModel(
            llm,
            base_url=base_url,
            api_key=api_key
        )