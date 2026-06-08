from google import genai
from tools import *
import json
import re
import time
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = {}

class FinalAnswer(BaseModel):
    tool: Optional[str] = None
    answer: str

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise Exception("Missing API_KEY in .env")

client = genai.Client(api_key=API_KEY)

tools_map = {
    "get_cpu_usage": get_cpu_usage,
    "get_memory_usage": get_memory_usage,
    "get_disk_usage": get_disk_usage,
}

SYSTEM_PROMPT = """
You are an AI agent.

You can call tools multiple times before giving a final answer.

If you need data:
Return only:
{
  "tool": "tool_name",
  "args": {}
}

When done:
{
  "tool": null,
  "answer": "final answer"
}

Available tools:
- get_cpu_usage
- get_memory_usage
- get_disk_usage

If user asks about computer health → always check all tools before final answer.
"""

def run_agent(question):

    messages = f"{SYSTEM_PROMPT}\n\nUser: {question}"
    
    max_steps = 5
    step = 0

    while step < max_steps:
        step += 1

        response = call_llm(messages)

        text = response.candidates[0].content.parts[0].text
        print("\n🤖 AI:", text)

        clean = extract_json(text)
        try:
            data = json.loads(clean)

            if data.get("tool") is not None:
                data = ToolCall.model_validate(data).model_dump()
            else:
                data = FinalAnswer.model_validate(data).model_dump()
        except Exception as e:
            print(f"⚠️ Invalid response: {e}")
            continue

        if data.get("tool") is not None:
            tool_name = data.get("tool")
            if tool_name is None:
                print("⚠️ No tool returned")
                continue
           
            if tool_name not in tools_map:
                print(f"⚠️ Unknown tool: {tool_name}")
                continue

            result = tools_map[tool_name](**data.get("args", {}))

            print("📦 Tool result:", result)

            messages += f"\nObservation: {result}"

        else:
            print("\n✅ FINAL:", data["answer"])
            return

def extract_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    return text.strip()

def call_llm(messages):
    for i in range(5):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages
            )
        except Exception as e:
            err = str(e)

            if "429" in err:
                wait_time = 60 

                import re
                match = re.search(r"retryDelay': '(\d+)", err)
                if match:
                    wait_time = int(match.group(1))

                print(f"⏳ Rate limit (429) - waiting {wait_time}s... [{i+1}/5]")
                time.sleep(wait_time)
                continue

            if "503" in err:
                wait_time = (2 ** i)
                print(f"⚠️ Server busy (503) - retry {i+1}/5 in {wait_time:.1f}s")
                time.sleep(wait_time)
                continue

            print(f"❌ Unexpected error: {type(e).__name__}")
            raise

    raise Exception("LLM failed after retries")

if __name__ == "__main__":
    print("=== AGENT START ===")
    run_agent("How is my computer doing?")
    # run_agent("How is my Memory doing?")
    # run_agent("How is my Memory doing?")
    # run_agent("How is my computer doing?")
    # run_agent("How is my disk doing?")
    # run_agent("Say hello")

    print("=== DONE ===")
