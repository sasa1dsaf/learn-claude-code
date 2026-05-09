#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
"""
s05_skill_loading.py - Skills

Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only
    |   - code-review: Review code...      |
    +--------------------------------------+

    When model calls load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

Key insight: "Don't put everything in the system prompt. Load on demand."
"""

import os
import re
import subprocess
import yaml
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"


# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --
class SkillLoader:
    """
    技能加载器：自动扫描、解析、管理所有技能文件夹下的 SKILL.md
    每个技能 = 一个目录 + 一个带YAML头的markdown文件
    提供两层能力：
    1）get_descriptions：给系统提示词用的技能列表
    2）get_content：获取某个技能的完整内容
    """

    def __init__(self, skills_dir: Path):
        """初始化：指定技能目录，自动加载所有技能"""
        self.skills_dir = skills_dir  # 技能根目录，如 ./skills
        self.skills = {}  # 内存存储所有技能：{技能名: {meta, body, path}}
        self._load_all()  # 自动加载所有技能

    def _load_all(self):
        """内部方法：递归扫描所有 SKILL.md 文件并加载"""
        # 如果技能目录不存在，直接返回
        if not self.skills_dir.exists():
            return

        # 递归找到所有子目录下的 SKILL.md
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()  # 读取文件全文

            # 拆分成 YAML 头部元信息 + 正文
            meta, body = self._parse_frontmatter(text)

            # 技能名优先用 YAML 里的 name，没有就用文件夹名
            name = meta.get("name", f.parent.name)

            # 把技能存入内存
            self.skills[name] = {
                "meta": meta,  # YAML 头：name, description, tags...
                "body": body,  # 技能正文（教程/规则/提示词）
                "path": str(f)  # 文件路径（方便调试）
            }

    def _parse_frontmatter(self, text: str) -> tuple:
        """解析 MD 文件的 YAML 头（--- 包裹的部分）"""
        # 正则匹配：开头 --- ... --- 后面是正文
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        # 没有 YAML 头 → 返回空meta + 原文
        if not match:
            return {}, text

        try:
            # 解析 YAML 为字典
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            # 解析失败 → 空字典
            meta = {}

        # 返回 (meta字典, 正文内容)
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        第一层能力：给系统提示词用 → 简短技能列表（名字+描述）
        让 LLM 知道自己有哪些技能可用
        """
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"  # 有标签就追加显示
            lines.append(line)

        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        第二层能力：LLM 调用 load_skill 后 → 返回完整技能内容
        用 <skill> 标签包裹，方便模型识别这是技能文档
        """
        skill = self.skills.get(name)

        # 技能不存在 → 返回错误
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"

        # 返回完整技能正文
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# ====================== 全局单例：整个程序共用一个技能加载器 ======================
SKILL_LOADER = SkillLoader(SKILLS_DIR)

# ====================== 系统提示词：把技能列表注入给模型 ======================
# Layer 1：把技能列表注入系统提示，让模型知道自己会什么
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""

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
        lines = safe_path(path).read_text().splitlines()
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
    # Layer 2：加载完整技能内容（模型调用该工具后获取完整技能教程）
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
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
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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
