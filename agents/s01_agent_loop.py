#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - AI 智能体核心循环

AI 编程智能体的核心原理，就一个模式：

    while stop_reason == "tool_use":
        调用大模型
        执行工具
        把结果返回给模型

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   工具执行结果  |
                          +---------------+
                          循环继续

这就是核心：把工具执行结果喂回给模型
直到模型决定停止。工业级的 Agent 只是在此基础上
增加策略、钩子、生命周期控制等。
"""

import os
import subprocess

# 解决 macOS 终端输入中文/退格问题（可忽略）
try:
    import readline

    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

# 导入 Anthropic SDK（兼容 Claude / 千问 / GLM / Kimi / DeepSeek）
from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 配置文件
load_dotenv(override=True)

# 如果配置了 BASE_URL，就去掉ANTHROPIC的默认认证（兼容中转/国内模型）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 创建 LLM 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 从环境变量读取模型 ID（如 qwen3.6-plus）
MODEL = os.environ["MODEL_ID"]

# ====================== 【修改 1】系统提示词：PowerShell ======================
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use Windows PowerShell commands. Act, don't explain."

# 定义工具列表：只给 AI 一个 bash 命令执行工具
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"}
        },
        "required": ["command"]
    }
}]


# 执行 bash 命令的安全封装函数
def run_bash(command: str) -> str:
    # ====================== 【修改 2】危险命令列表（PowerShell / Windows） ======================
    dangerous = [
        "Remove-Item -Recurse -Force C:\\",
        "rd /s /q C:\\",
        "Start-Process powershell -Verb RunAs",
        "shutdown",
        "restart",
        "Stop-Computer",
        "Restart-Computer",
        "Set-ItemProperty",
        "New-ItemProperty",
        "Remove-ItemProperty",
        "format",
        "diskpart"
    ]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # 执行 shell 命令，捕获输出
        r = subprocess.run(
            command, shell=True, cwd=os.getcwd(),
            capture_output=True, text=True, timeout=120
        )
        # 合并 stdout + stderr
        out = (r.stdout + r.stderr).strip()

        # 限制输出长度，避免超长
        return out[:50000] if out else "(no output)"

    # 超时/异常处理
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ====================== 核心：AI Agent 循环 ======================
def agent_loop(messages: list):
    # 无限循环：思考 → 执行工具 → 再思考
    while True:
        # 1. 调用大模型（带工具能力）
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 2. 把模型的回复存入历史
        messages.append({"role": "assistant", "content": response.content})

        # 3. 如果模型不调用工具 → 结束循环
        if response.stop_reason != "tool_use":
            return

        # 4. 如果模型要调用工具 → 遍历所有工具调用块
        results = []
        for block in response.content:

            if block.type == "tool_use":
                # 打印 AI 要执行的命令（黄色）
                print(f"\033[33m$ {block.input['command']}\033[0m")

                # 执行命令
                output = run_bash(block.input["command"])

                # 打印输出（前200字符）
                print(output[:200])

                # 构造工具执行结果，返回给模型
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })

        # 5. 把工具结果喂回给 AI，让它继续思考
        messages.append({"role": "user", "content": results})


# ====================== 命令行交互入口 ======================
if __name__ == "__main__":

    history = []  # 保存对话历史
    while True:
        try:
            # 命令行输入提示（蓝色）
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 输入 q / exit 退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户问题加入历史
        history.append({"role": "user", "content": query})

        # 运行智能体循环
        agent_loop(history)

        # 打印最终回答
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()