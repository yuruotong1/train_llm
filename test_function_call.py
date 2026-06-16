"""测试 MiniMind OpenAI 兼容服务的 tool call 能力。

前置条件: 先启动服务
    uv run python trainer/server.py --stage dpo

运行:
    uv run python test_function_call.py
"""

import json
import sys

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8998"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如 北京"},
                },
                "required": ["city"],
            },
        },
    }
]


def fake_get_weather(city: str) -> str:
    return json.dumps({"city": city, "weather": "晴", "temperature": "26°C"}, ensure_ascii=False)


def call_chat(messages, tools=None):
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "minimind",
            "messages": messages,
            "tools": tools or [],
            "temperature": 0.7,
            "max_tokens": 256,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    health = requests.get(f"{BASE_URL}/health", timeout=10).json()
    print(f"服务状态: {health}")

    messages = [{"role": "user", "content": "北京今天天气怎么样？"}]
    print("\n[第一轮] 发送用户问题 + tools 定义...")
    result = call_chat(messages, tools=TOOLS)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls")

    if not tool_calls:
        print("\n模型没有触发 tool_call，直接回答:")
        print(message.get("content"))
        return

    print(f"\n模型触发了 {len(tool_calls)} 个 tool_call:")
    messages.append(message)
    for call in tool_calls:
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
        print(f"  - {name}({arguments})")

        if name == "get_weather":
            tool_result = fake_get_weather(**arguments)
        else:
            tool_result = json.dumps({"error": f"未知函数 {name}"}, ensure_ascii=False)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result,
            }
        )

    print("\n[第二轮] 把工具执行结果喂回模型，获取最终回答...")
    final = call_chat(messages, tools=TOOLS)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print("\n最终回答:")
    print(final["choices"][0]["message"].get("content"))


if __name__ == "__main__":
    main()
