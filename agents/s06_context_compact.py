#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.

"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace non-read_file tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
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

# 上下文压缩配置
THRESHOLD = 500               # 自动压缩阈值：超过 50000 token 触发 Layer2
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 完整对话历史存档目录
KEEP_RECENT = 3                  # Layer1：保留最近 3 条工具结果不压缩
PRESERVE_RESULT_TOOLS = {"read_file"}  # 不压缩的工具（read_file 要保留原文）
COMPACT_MIN_CONTENT_LEN = 100  # 👈 新加：统一配置压缩最小长度

def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数量：按 ~4 个字符 = 1 个 token 计算"""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact 轻量压缩 --
# 作用：每轮都悄悄执行，把太久远的工具结果替换成简短占位符
def micro_compact(messages: list) -> list:
    """
    Layer 1 轻量压缩：
    统一字典结构判断，一次遍历完成两件事：
    1) 收集所有 tool_result
    2) 构建 tool_use_id → tool_name 映射
    彻底消除 dict / Pydantic 实例混用问题
    """
    tool_results = []
    tool_name_map = {}

    # 👇 一次遍历，同时完成两件事，效率更高
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", [])

        # 只处理列表格式的 content
        if not isinstance(content, list):
            continue

        # --------------------------
        # 1. 收集 assistant 的 tool_use（构建映射表）
        # --------------------------
        if role == "assistant":
            for block in content:
                # 👇 统一按字典判断，不再用 hasattr / Pydantic 判断
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    tool_name = block.get("name")
                    if tool_id and tool_name:
                        tool_name_map[tool_id] = tool_name

        # --------------------------
        # 2. 收集 user 的 tool_result
        # --------------------------
        if role == "user":
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    # 不足保留数量，直接返回
    if len(tool_results) <= KEEP_RECENT:
        return messages

    # --------------------------
    # 压缩旧的工具结果（保留最近 N 条）
    # --------------------------
    to_clear = tool_results[:-KEEP_RECENT]

    for _, _, result in to_clear:
        content = result.get("content", "")
        # 👇 使用常量，不再硬编码 100
        if not isinstance(content, str) or len(content) <= COMPACT_MIN_CONTENT_LEN:
            continue

        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")

        # 白名单工具不压缩
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        # 替换为精简占位符
        result["content"] = f"[Previous: used {tool_name}]"

    return messages


# -- Layer 2: auto_compact 自动重度压缩 --
# 作用：上下文超阈值时触发 → 存档完整历史 → LLM 总结 → 只保留总结
def auto_compact(messages: list, focus: str = "") -> list:  # 👈 加了 focus 参数
    # 1. 把完整对话历史保存到磁盘（信息不丢失）
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    # 2. 截取最后 80000 字符交给 LLM 做总结（避免超限）
    conversation_text = json.dumps(messages, default=str)[-80000:]

    # 3. 调用模型生成总结：支持 focus 聚焦
    base_prompt = (
        "Summarize this conversation for continuity. Include:\n"
        "1) What was accomplished\n2) Current state\n3) Key decisions made\n"
        "Be concise but preserve critical details."
    )

    # 👇 关键：把 focus 加进提示词
    if focus.strip():
        base_prompt += f"\n\nIMPORTANT FOCUS FOR SUMMARY: {focus.strip()}"

    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{base_prompt}\n\n{conversation_text}"}],
        max_tokens=2000,
    )

    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    if not summary:
        summary = "No summary generated."

    # 4. 用【单条总结】替换全部历史消息
    messages[:] = [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]
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
        # 强制使用 UTF-8 读取，彻底解决 GBK 报错
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
    # Layer 3：手动压缩工具
    # 模型调用这个工具 → 触发 full 总结压缩（和 auto_compact 一样）
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


# -- 主 Agent 循环：整合三层压缩 --
def agent_loop(messages: list):
    while True:
        # Layer 1 -- micro_compact: 每次 LLM 调用前, 将旧的 tool result 替换为占位符。
        micro_compact(messages)

        # Layer 2 -- auto_compact: token 超过阈值时, 保存完整对话到磁盘, 让 LLM 做摘要。
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            auto_compact(messages)

        # 调用模型生成回复
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        # 如果模型没有调用工具，直接结束循环
        if response.stop_reason != "tool_use":
            return

        # 执行模型调用的工具
        results = []

        for block in response.content:
            if block.type == "tool_use":
                # Layer 3 -- manual compact: compact 工具按需触发同样的压缩机制（auto_compact）。
                if block.name == "compact":
                    # 获取 focus 参数
                    focus = block.input.get("focus", "")
                    print(f"[manual compact | focus: {focus}]")
                    # 直接执行压缩
                    auto_compact(messages, focus=focus)
                    # 构造 tool_result
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Compressed successfully! Focus: {focus or 'default'}"
                    })
                    # 结束本轮，不再执行其他工具
                    messages.append({"role": "user", "content": results})
                    return

                # 普通工具逻辑
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"

                print(f"\033[33m> {block.name}:\033[0m")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        # 把工具结果加入上下文
        messages.append({"role": "user", "content": results})

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
