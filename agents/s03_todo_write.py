#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

模型通过 TodoManager 管理自己的任务进度，忘记更新时会被强制提醒。

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

核心思想：智能体可以自己跟踪任务进度，开发者也能清晰看到执行状态。
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量配置
load_dotenv(override=True)

# 如果配置了 BASE_URL，清除不需要的认证变量（适配国内模型中转）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 工作目录 = 当前运行目录
WORKDIR = Path.cwd()

# 初始化 LLM 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词：告诉模型要使用 todo 管理任务、优先使用工具
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# ====================== Todo 任务管理器：让模型不乱跑、不跳步、不忘事 ======================
class TodoManager:
    def __init__(self):
        self.items = []  # 存储所有任务列表

    # 更新任务列表（AI 调用 todo 工具时执行）
    def update(self, items: list) -> str:
        # 限制最多 20 个任务，防止上下文爆炸
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0

        # 校验每一条任务的格式、状态是否合法
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 任务内容不能为空
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            # 状态只能是 pending / in_progress / completed
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            # 统计正在进行中的任务数量
            if status == "in_progress":
                in_progress_count += 1

            validated.append({"id": item_id, "text": text, "status": status})

        # 强制规则：同一时间只能有一个任务进行中 → 保证顺序执行
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    # 把任务列表渲染成人类可读的格式
    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 统计完成率
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


# 创建全局 todo 实例
TODO = TodoManager()


# ====================== 工具函数：安全路径、文件操作、命令执行 ======================

# 安全路径检查：防止文件逃逸出工作区
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

# 执行 shell 命令（带危险命令拦截）
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

# 读取文件
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

# 写入文件
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

# 编辑文件（替换指定文本）
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


# ====================== 工具映射表：AI 调用的工具名 → 对应执行函数 ======================
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),  # 任务管理工具
}

# 工具定义（给 AI 看的接口描述）
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# ====================== 核心智能体循环：带 todo 强制提醒 ======================
def agent_loop(messages: list):
    # 记录连续多少轮没有更新 todo
    rounds_since_todo = 0

    while True:
        # 调用大模型
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 把模型回复存入历史
        messages.append({"role": "assistant", "content": response.content})

        # 如果不调用工具，结束循环
        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False  # 标记本轮是否调用了 todo

        # 执行所有工具调用
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"

                # 打印工具执行日志
                print(f"> {block.name}:")
                print(str(output)[:200])

                # 把结果返回给模型
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

                # 如果用了 todo，重置计数
                if block.name == "todo":
                    used_todo = True

        # 更新连续未使用 todo 的轮数
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        # # 如果连续 3 轮没更 todo → 强制注入提醒
        # if rounds_since_todo >= 3:
        #     results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})

        # 改进：
        # 1. LLM 不调用 todo → 任务列表不会自动刷新在上下文中
        # 2. 三轮没有调用TODO，只提醒 “更新任务” → 旧任务上下文被工具输出淹没，LLM 会幻觉
        # 3. 提醒 + 把内存任务一起注入 → 让内存里的任务有了价值，能真正反向作用于 LLM
        # 4. 形成完美闭环，任务进度不丢失、不幻觉、不跑偏
        if rounds_since_todo >= 3:
            # 直接把内存里存的真实任务 渲染出来 塞回去！
            todo_list_text = TODO.render()

            # 强制提醒 + 标准答案一起给
            reminder = f"""
        <reminder>YOU MUST UPDATE YOUR TASK STATUS!
        Below is YOUR REAL TASK LIST (from memory):

        {todo_list_text}

        Please update the status using the todo tool NOW.</reminder>
            """
            results.append({"type": "text", "text": reminder})

        # 把工具结果 + 提醒 喂给模型
        messages.append({"role": "user", "content": results})


# ====================== 命令行交互入口 ======================
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印最终回答
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()