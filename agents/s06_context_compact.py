#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.

"""
s06_context_compact.py - Compact
Three-layer compression pipeline so the agent can work forever.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (f"You are a coding agent working at {WORKDIR}."
          f"- For system commands, ONLY use PowerShell commands (do NOT use Linux commands like ls, use dir instead).)"
          f"- Use tools to solve tasks correctly.")

# 上下文压缩配置（测试用，阈值设很小）
THRESHOLD = 8000               # 测试：很小就触发 Layer2
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 1
PRESERVE_RESULT_TOOLS = {"read_file"}
COMPACT_MIN_CONTENT_LEN = 100

def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数量：按 ~4 个字符 = 1 个 token 计算"""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact 轻量压缩 --
def micro_compact(messages: list) -> list:
    print("\n=== 🔍 Layer1: micro_compact 运行中 ===")  # 调试：确认进入Layer1
    tool_results = []
    tool_name_map = {}

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", [])

        if not isinstance(content, list):
            continue

        # 1. 收集 assistant 的 tool_use（对象）
        if role == "assistant":
            for block in content:
                if hasattr(block, "type") and block.type == "tool_use":
                    print(f"ℹ️ Layer1：增加{block.id}到{block.name}的映射")
                    tool_name_map[block.id] = block.name

        # 2. 收集 user 的 tool_result（字典）
        if role == "user":
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    if len(tool_results) <= KEEP_RECENT:
        print(f"ℹ️ Layer1：tool_result数量({len(tool_results)}) ≤ {KEEP_RECENT}，无需压缩")
        return messages

    to_clear = tool_results[:-KEEP_RECENT]
    print(f"ℹ️ Layer1：待压缩旧tool_result数量：{len(to_clear)}")

    for _, _, result in to_clear:
        content = result.get("content", "")
        if not isinstance(content, str) or len(content) <= COMPACT_MIN_CONTENT_LEN:
            continue

        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")

        if tool_name in PRESERVE_RESULT_TOOLS:
            print(f"ℹ️ Layer1：白名单工具 {tool_name} 不压缩")
            continue

        result["content"] = f"[Previous: used {tool_name}]"
        print(f"✅ Layer1：已压缩 → {tool_name}")

    return messages


# -- Layer 2: auto_compact 自动重度压缩 --
def auto_compact(messages: list, focus: str = "") -> list:
    print("\n=== 🚀 Layer2: auto_compact 执行 ===")  # 调试：确认进入Layer2
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    print(f"📝 Layer2：已保存完整历史 → {transcript_path}")

    conversation_text = json.dumps(messages, default=str)[-80000:]

    base_prompt = (
        "Summarize this conversation for continuity. Include:\n"
        "1) What was accomplished\n2) Current state\n3) Key decisions made\n"
        "Be concise but preserve critical details."
    )
    if focus.strip():
        base_prompt += f"\n\nIMPORTANT FOCUS FOR SUMMARY: {focus.strip()}"

    print("🧠 Layer2：请求LLM生成摘要...")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{base_prompt}\n\n{conversation_text}"}],
        max_tokens=2000,
    )

    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."
    print(f"📄 Layer2：生成摘要完成（长度：{len(summary)}）")

    messages[:] = [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]
    print(f"✅ Layer2：压缩完成，当前消息条数：{len(messages)}")
    return messages


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


# -- 主 Agent 循环：整合三层压缩 --
def agent_loop(messages: list):
    while True:
        # 全局状态打印（最关键验证：看token和消息条数）
        print("\n================================")
        print(f"📊 当前token估算：{estimate_tokens(messages)}")
        print(f"📬 当前消息条数：{len(messages)}")
        print(f"⚡ 压缩阈值：{THRESHOLD}")
        print("================================\n")

        # Layer1
        micro_compact(messages)

        # Layer2
        if estimate_tokens(messages) > THRESHOLD:
            print(f"⚠️ 超过阈值！触发auto_compact")
            auto_compact(messages)

        # 调用模型
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        # 执行工具
        results = []
        manual_compact = None  # 👈 标记是否需要 compact

        # 第一步：只标记manual_compact，先执行所有普通工具
        for block in response.content:
            if block.type == "tool_use":
                # Layer 3：模型主动调用 compact：先标记，不立即执行
                if block.name == "compact":
                    manual_compact = block
                    continue

                # 第一步：普通工具正常执行
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"\033[33m> {block.name}: {block.input}\033[0m")
                print(str(output)[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)
                })

        # 把普通工具结果先加进去
        if results:
            messages.append({"role": "user", "content": results})

        # 第二步：最后执行 compact（如果有）
        if manual_compact:
            block = manual_compact
            print("\n=== 🛠 Layer3: 模型调用compact工具 ===")
            focus = block.input.get("focus", "")
            print(f"🎯 Layer3 focus：{focus or '无'}")
            auto_compact(messages, focus=focus)
            return  # 最后再退出

"""
疑问：为什么不能像 bash /read_file 那样在 handler 里直接执行 auto_compact？
解答：因为普通工具只需要零散参数（path、command）；压缩需要完整的全局对话列表 messages，
     LLM 无法把内存变量 messages 传入工具函数、TOOL_HANDLERS 拿不到 messages
 👉 所以只能设计成：LLM 只发一个「要压缩」的信号 → 外层 agent_loop 拿到信号 
    → 自己动手操作 messages 做压缩
"""

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()