import argparse
import json
import re
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextStreamer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.runtime import ensure_workspace
from core.train_lib import load_stage_for_inference


app = FastAPI(title="MiniMind OpenAI Server")
MODEL = None
TOKENIZER = None
SERVER_STAGE = "dpo"


class ChatRequest(BaseModel):
    model: str = "minimind"
    messages: list
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: dict | None = None

    def thinking_enabled(self) -> bool:
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return bool(
                self.chat_template_kwargs.get("open_thinking")
                or self.chat_template_kwargs.get("enable_thinking")
            )
        return False


class QueueStreamer(TextStreamer):
    def __init__(self, tokenizer, queue: Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text: str):
    reasoning_content = None
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning_content = parts[0].replace("<think>", "").strip()
        text = parts[1].strip()

    tool_calls = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    for idx, payload in enumerate(matches):
        try:
            call = json.loads(payload.strip())
            tool_calls.append(
                {
                    "id": f"call_{int(time.time())}_{idx}",
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
            )
        except Exception:
            continue
    if tool_calls:
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def _build_prompt(messages, tools=None, open_thinking=False):
    return TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools or None,
        open_thinking=open_thinking,
    )


def _generate_text(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    prompt = _build_prompt(messages, tools=tools, open_thinking=open_thinking)
    device = next(MODEL.parameters()).device
    inputs = TOKENIZER(prompt, return_tensors="pt").to(device)
    generated = MODEL.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=TOKENIZER.eos_token_id,
    )
    answer_ids = generated[0][inputs.input_ids.shape[1]:]
    return TOKENIZER.decode(answer_ids, skip_special_tokens=True)


def _generate_stream(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    prompt = _build_prompt(messages, tools=tools, open_thinking=open_thinking)
    device = next(MODEL.parameters()).device
    inputs = TOKENIZER(prompt, return_tensors="pt").to(device)
    queue = Queue()
    streamer = QueueStreamer(TOKENIZER, queue)

    def _worker():
        MODEL.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=TOKENIZER.eos_token_id,
            streamer=streamer,
        )

    Thread(target=_worker, daemon=True).start()

    full_text = ""
    emitted = 0
    thinking_ended = not open_thinking

    while True:
        chunk = queue.get()
        if chunk is None:
            break
        full_text += chunk

        if not thinking_ended:
            pos = full_text.find("</think>")
            if pos >= 0:
                thinking_ended = True
                new_reason = full_text[emitted:pos]
                if new_reason:
                    yield json.dumps({"choices": [{"delta": {"reasoning_content": new_reason}}]}, ensure_ascii=False)
                emitted = pos + len("</think>")
                tail = full_text[emitted:].lstrip("\n")
                emitted = len(full_text) - len(tail)
                if tail:
                    yield json.dumps({"choices": [{"delta": {"content": tail}}]}, ensure_ascii=False)
                    emitted = len(full_text)
            else:
                new_reason = full_text[emitted:]
                if new_reason:
                    yield json.dumps({"choices": [{"delta": {"reasoning_content": new_reason}}]}, ensure_ascii=False)
                    emitted = len(full_text)
        else:
            new_content = full_text[emitted:]
            if new_content:
                yield json.dumps({"choices": [{"delta": {"content": new_content}}]}, ensure_ascii=False)
                emitted = len(full_text)

    _, _, tool_calls = parse_response(full_text)
    if tool_calls:
        yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
    yield json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]},
        ensure_ascii=False,
    )


@app.get("/health")
def health():
    return {"status": "ok", "stage": SERVER_STAGE}


@app.get("/v1/models")
def models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"id": f"minimind-{SERVER_STAGE}", "object": "model", "owned_by": "local"},
                {"id": "minimind", "object": "model", "owned_by": "local"},
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in _generate_stream(
                    request.messages,
                    request.temperature,
                    request.top_p,
                    request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.thinking_enabled(),
                )),
                media_type="text/event-stream",
            )

        raw_answer = _generate_text(
            request.messages,
            request.temperature,
            request.top_p,
            request.max_tokens,
            tools=request.tools,
            open_thinking=request.thinking_enabled(),
        )
        content, reasoning_content, tool_calls = parse_response(raw_answer)
        message = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls

        return JSONResponse(
            {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    global MODEL, TOKENIZER, SERVER_STAGE
    ensure_workspace()
    parser = argparse.ArgumentParser(description="MiniMind OpenAI-compatible server")
    parser.add_argument("--stage", default="dpo", choices=["pretrain", "sft", "dpo"], help="默认加载 dpo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    args = parser.parse_args()

    SERVER_STAGE = args.stage
    MODEL, TOKENIZER = load_stage_for_inference(args.stage)
    print(f"服务已加载 {args.stage} checkpoint")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
