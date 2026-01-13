import asyncio
import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from zhipu_model import ZhipuModel
from dotenv import load_dotenv

load_dotenv()

class CityInfo(BaseModel):
    city: str = Field(description="The name of the city")
    country: str = Field(description="The country the city is in")
    population: int = Field(description="Approximate population")

async def main():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("ZHIPUAI_API_KEY or LLM_API_KEY not found in .env, skipping test.")
        return

    model = ZhipuModel(model_name="glm-4.7", api_key=api_key)
    
    agent = Agent(model, result_type=CityInfo, system_prompt="You are a helpful assistant.")

    try:
        result = await agent.run("Tell me about Paris")
        print("Result:", result.data)
        print("Usage:", result.usage())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
