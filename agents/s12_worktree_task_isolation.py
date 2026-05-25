#!/usr/bin/env python3
# Harness: directory isolation -- parallel execution lanes that never collide.
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.

    .tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }

    .worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }

Key insight: "Isolate by directory, coordinate by task ID."
"""

#!/usr/bin/env python3
# Harness: directory isolation -- parallel execution lanes that never collide.
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation
功能：基于 Git Worktree + 任务体系 实现目录级隔离，支撑并行任务执行
设计分层：
  1. 任务(Task)：控制面，负责任务生命周期、状态流转、绑定工作树
  2. 工作树(Worktree)：执行面，独立目录环境，任务在此隔离运行
数据存储结构：
  .tasks/task_12.json      单个任务持久化文件
  .worktrees/index.json    所有工作树统一索引清单
  .worktrees/events.jsonl 全量生命周期事件日志

最终的工作目录：
你的项目仓库（Git 根目录）
├── .git/               # Git 本体（只有一份！共享给所有工作树）
├── 正常项目文件        # 主分支代码
├── main.py
├── src/
└── ...

├── .tasks/             # 任务管理（脚本创建）
│   ├── task_1.json     # 任务1
│   ├── task_2.json     # 任务2
│   └── task_3.json
│
└── .worktrees/         # 工作树总目录（脚本核心）
    ├── index.json      # 工作树索引清单
    ├── events.jsonl    # 生命周期日志
    │
    ├── feature-login/        # 工作树 1 → 分支 wt/feature-login
    │   ├── 项目文件
    │   ├── main.py
    │   └── src/
    │
    ├── feature-pay/          # 工作树 2 → 分支 wt/feature-pay
    │   ├── 项目文件
    │   └── src/
    │
    └── refactor-docs/        # 工作树 3 → 分支 wt/refactor-docs
        ├── 项目文件
        └── src/

核心设计思想：Isolate by directory, coordinate by task ID.
（目录做环境隔离，任务ID做多组件协同）
第一层：仓库 ↔ 分支
plaintext
一个云端仓库
   ├── 分支 main
   ├── 分支 feature/A
   ├── 分支 feature/B
   └── 分支 feature/C
一个仓库 = 可以有很多分支
分支 = 仓库的一个 “独立版本”
第二层：分支 ↔ 工作树
plaintext
一个本地分支
   ├── 主工作区（默认只能有1个）
   └── worktree 工作区1（独立目录）
       worktree 工作区2（独立目录）
       worktree 工作区3（独立目录）
一个分支 = 可以挂在多个工作树上
工作树 = 分支的一个 “独立运行目录”
终极套娃关系（完美对应）
仓库 ↔ 分支
= 大容器里放多个独立版本
分支 ↔ 工作树
= 一个版本里放多个独立运行目录
所以：
工作树 是 分支的 “实例化目录”
分支 是 仓库的 “实例化版本”
"""


"""
- 问题：为什么不能直接在提示词告诉每个agent，执行任务前需要用git命令把当前仓库挂载到一个新工作树上呢，
    毕竟给agent提供了bash工具啊?
- 解答：Agent 确实可以自己执行 bash 命令，但真实工程里绝对不会这么做，原因如下：
    1. Agent不稳定、不可控、无状态，再强也是概率模型，会遗忘会幻觉，
    Git worktree是底层能力，不能给 Agent 直接用，一旦执行错误命令会污染主分支，太危险了，
    必须用一层 Python 管理系统把它包起来，变成安全、可控、可追踪的能力！
    2. Agent 没有事务、没有日志、没有回滚，没有全局状态感知：
    Agent 不知道已经有哪些工作树、哪个任务对应哪个目录，它是瞎子，不能让瞎子自己管理 Git 工作树。

"""

# 导入标准库
import json
import os
import re
import subprocess
import time
from pathlib import Path

# 导入第三方库：大模型客户端 + 环境变量加载
from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 环境变量，强制覆盖已有环境变量
load_dotenv(override=True)

# 若配置了自定义大模型接口地址，清除原有认证令牌，避免冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 定义全局工作目录 = 当前脚本运行目录
WORKDIR = Path.cwd()
# 初始化 Anthropic 大模型客户端，使用自定义接口地址
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 从环境变量读取要使用的大模型名称
MODEL = os.environ["MODEL_ID"]


def detect_repo_root(cwd: Path) -> Path | None:
    """
    检测当前目录所属的 Git 仓库根目录
    :param cwd: 待检测目录路径
    :return: 仓库根目录Path对象 / 非Git仓库则返回 None
    """
    try:
        # 执行git命令：获取当前仓库顶层根目录
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,                # 指定命令执行目录
            capture_output=True,    # 捕获标准输出、标准错误
            text=True,              # 输出转为字符串而非字节
            timeout=10,             # 命令10秒超时保护
        )
        # 命令执行失败（不在Git仓库内）
        if r.returncode != 0:
            return None
        # 清洗路径并校验目录是否真实存在
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        # 任意异常（命令不存在、权限等）统一返回None
        return None

# 全局仓库根目录：识别到Git仓库则用仓库根，否则使用当前工作目录
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

# 大模型系统提示词：定义智能代理的行为规则与工具使用规范
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)


# -- EventBus: 仅追加式事件总线，用于全链路生命周期观测、日志审计 --
class EventBus:
    def __init__(self, event_log_path: Path):
        self.path = event_log_path          # 事件日志文件路径（jsonl格式）
        # 递归创建日志所在目录，已存在则忽略
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 日志文件不存在则创建空文件
        if not self.path.exists():
            self.path.write_text("")
    """
    events.jsonl（生命周期日志）
    格式：jsonl = 一行一条 JSON 事件
    作用：记录工作树 / 任务什么时候创建、删除、失败……
    
    文件内容：
    {"event":"worktree.create.before","ts":1745678901.123,"task":{"id":1},"worktree":{"name":"auth-refactor","base_ref":"HEAD"}}
    {"event":"worktree.create.after","ts":1745678902.456,"task":{"id":1},"worktree":{"name":"auth-refactor","path":"/...","branch":"wt/auth-refactor","status":"active"}}
    {"event":"worktree.remove.before","ts":1745678999.123,"task":{"id":3},"worktree":{"name":"docs-update","path":"/..."}}
    {"event":"task.completed","ts":1745678999.456,"task":{"id":3,"subject":"更新文档","status":"completed"},"worktree":{"name":"docs-update"}}
    {"event":"worktree.remove.after","ts":1745679000.123,"task":{"id":3},"worktree":{"name":"docs-update","path":"/...","status":"removed"}}
    
    字段含义：
    event：事件名
    worktree.create.before
    worktree.create.after
    worktree.create.failed
    worktree.remove.before
    worktree.remove.after
    task.completed
    ts：时间戳
    task：关联任务
    worktree：关联工作树
    error：可选，报错信息
    """
    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        """
        写入一条生命周期事件（追加写入，不可修改历史日志）
        :param event: 事件名称（如 worktree.create、task.completed）
        :param task: 关联的任务信息字典
        :param worktree: 关联的工作树信息字典
        :param error: 异常信息，正常流程传None
        """
        # 组装日志载体
        payload = {
            "event": event,
            "ts": time.time(),      # 时间戳（Unix 时间）
            "task": task or {},     # 无任务则存空字典
            "worktree": worktree or {},
        }
        # 存在异常则追加错误字段
        if error:
            payload["error"] = error
        # 追加写入一行JSON（jsonl标准格式）
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        """
        查询最近的N条事件日志
        :param limit: 最大返回条数，默认20条
        :return: 格式化后的JSON字符串
        """
        # 条数限制：最小1条，最大200条，防止读取日志过载
        n = max(1, min(int(limit or 20), 200))
        # 读取全部日志并按行分割
        lines = self.path.read_text(encoding="utf-8").splitlines()
        # 截取最后N条最新日志
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                # 解析单行JSON日志
                items.append(json.loads(line))
            except Exception:
                # 解析失败，标记为解析错误日志
                items.append({"event": "parse_error", "raw": line})
        # 格式化输出JSON
        return json.dumps(items, indent=2)


# -- TaskManager: 持久化任务管理器，实现任务看板、状态流转、绑定/解绑工作树 --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir                # 任务文件存储根目录 .tasks/
        self.dir.mkdir(parents=True, exist_ok=True)
        # 初始化下一个可用任务ID = 当前已有最大ID + 1
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """私有方法：扫描目录，获取当前已存在任务的最大ID"""
        ids = []
        # 匹配所有 task_*.json 任务文件
        for f in self.dir.glob("task_*.json"):
            try:
                # 从文件名提取数字ID：task_12.json -> 12
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                continue
        # 无任务返回0，否则返回最大ID
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        """私有方法：根据任务ID拼接对应JSON文件路径"""
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        """私有方法：加载指定ID的任务数据，不存在则抛异常"""
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """私有方法：将任务字典持久化写入JSON文件"""
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """
        创建一条新任务
        :param subject: 任务标题/主题
        :param description: 任务详细描述（可选）
        :return: 新任务JSON字符串
        """
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",        # 任务状态：pending=待处理
            "owner": "",                # 任务负责人
            "worktree": "",             # 绑定的工作树名称，初始为空
            "blockedBy": [],            # 前置依赖任务列表
            "created_at": time.time(),  # 创建时间戳
            "updated_at": time.time(),  # 最后更新时间戳
        }
        self._save(task)
        # ID自增，为下一个任务做准备
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """根据任务ID查询任务详情"""
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        """判断指定ID的任务是否存在"""
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        """
        更新任务状态或负责人
        :param task_id: 任务ID
        :param status: 新状态（仅支持 pending / in_progress / completed）
        :param owner: 新负责人名称
        :return: 更新后的任务JSON
        """
        task = self._load(task_id)
        # 校验并更新任务状态
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        # 更新负责人
        if owner is not None:
            task["owner"] = owner
        # 刷新最后更新时间
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        """
        将任务与指定工作树绑定
        :param task_id: 任务ID
        :param worktree: 工作树名称
        :param owner: 追加设置任务负责人（可选）
        :return: 绑定后的任务JSON
        """
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        # 待处理任务绑定工作树后，自动转为「执行中」
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        """解除任务与工作树的绑定关系"""
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        """罗列所有任务，格式化展示状态、负责人、绑定工作树"""
        tasks = []
        # 按文件名排序读取所有任务
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            # 状态对应展示标记
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(t["status"], "[?]")
            # 拼接负责人、工作树后缀
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
        return "\n".join(lines)

# 全局单例：任务管理器、事件总线
TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


# -- WorktreeManager: Git Worktree 管理器，负责工作树增删查、命令执行、索引维护 --
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root      # Git仓库根目录
        self.tasks = tasks              # 关联全局任务管理器
        self.events = events            # 关联全局事件总线
        self.dir = repo_root / ".worktrees"  # 工作树存储根目录
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"  # 工作树总索引文件
        # 索引文件不存在则初始化空索引
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        # 检测当前环境是否可用Git命令
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        """私有方法：检测当前目录是否为合法Git仓库"""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        """
        私有通用Git命令执行封装
        :param args: git子命令+参数列表
        :return: 命令输出文本
        """
        # 无Git环境直接抛异常
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,  # Git操作超时120秒
        )
        # 命令执行失败，拼接输出信息抛异常
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        """私有方法：加载工作树总索引文件"""
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        """私有方法：写入更新后的工作树索引"""
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str) -> dict | None:
        """私有方法：根据工作树名称在索引中查找条目"""
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str):
        """私有方法：校验工作树名称合法性（字符、长度限制）"""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    """
        index.json（工作树索引清单）
        作用：记录所有工作树的信息，相当于 “工作树花名册”
        
        文件内容：
        {
            "worktrees": [
                {
                    "name": "auth-refactor",
                    "path": "/your/repo/.worktrees/auth-refactor",
                    "branch": "wt/auth-refactor",
                    "task_id": 1,
                    "status": "active",
                    "created_at": 1745678901.123456
                },
                {
                    "name": "payment-fix",
                    "path": "/your/repo/.worktrees/payment-fix",
                    "branch": "wt/payment-fix",
                    "task_id": 2,
                    "status": "kept",
                    "created_at": 1745678922.123456,
                    "kept_at": 1745679000.123456
                },
                {
                    "name": "docs-update",
                    "path": "/your/repo/.worktrees/docs-update",
                    "branch": "wt/docs-update",
                    "task_id": 3,
                    "status": "removed",
                    "created_at": 1745678955.123456,
                    "removed_at": 1745678999.123456
                }
            ]
        }
        
        字段含义:
        name：工作树名字
        path：真实目录路径
        branch：对应的 git 分支
        task_id：绑定的任务 ID
        status：状态
        active：正在用
        kept：保留
        removed：已删除
        created_at / kept_at / removed_at：时间戳
    """

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        """
        创建新Git工作树
        :param name: 工作树名称
        :param task_id: 绑定的任务ID（可选）
        :param base_ref: 基于哪个Git分支/提交创建，默认HEAD（当前分支）
        :return: 新工作树索引条目JSON
        """
        self._validate_name(name)
        # 同名工作树已存在则报错
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        # 指定任务ID时，校验任务必须存在
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name          # 工作树物理目录
        branch = f"wt/{name}"           # 自动生成关联分支名：wt/工作树名
        # 记录【创建前】事件
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        try:
            # 执行git worktree add：创建工作树 + 新建独立分支
            """
            Git 会自动做 3 件事：
            1. 创建文件夹.worktrees/你的工作树名/
            2. 把仓库当前分支的所有代码，完整拉一份到这个文件夹
            3. 切换到新分支（wt/xxx）
            """
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            # 组装工作树索引条目
            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",     # 状态：正常运行中
                "created_at": time.time(),
            }

            # 更新总索引
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            # 绑定对应任务
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            # 记录【创建完成】事件
            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            return json.dumps(entry, indent=2)
        except Exception as e:
            # 记录【创建失败】事件，再向外抛异常
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        """格式化罗列所有已托管的工作树"""
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        """查看指定工作树的git状态（文件变更、分支信息）
            e.g.
            ## main        → 当前在 main 分支
            M  app.py      → app.py 已修改（未提交）
            ?? temp.log    → temp.log 是新文件，未被追踪
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        # 在工作树目录执行 git status
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        """
        在指定工作树目录中执行Shell命令
        :param name: 工作树名称
        :param command: 待执行命令
        :return: 命令输出
        """
        # 高危命令黑名单拦截
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            # 在工作树目录执行shell命令
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,  # 命令超时5分钟
            )
            out = (r.stdout + r.stderr).strip()
            # 限制最大输出长度，避免日志溢出
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        """
        删除指定工作树
        :param name: 工作树名称
        :param force: 是否强制删除（忽略文件变更）
        :param complete_task: 是否同步将绑定任务标记为「已完成」
        :return: 执行结果文本
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        # 记录【删除前】事件
        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            # 组装 git worktree remove 参数
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)

            # 开启标记任务完成：更新任务状态 + 解绑工作树
            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                # 记录【任务完成】事件
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            # 更新索引：标记工作树状态为 removed，并记录删除时间
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            # 记录【删除完成】事件
            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            # 记录【删除失败】事件
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        """将工作树标记为「保留」状态（不删除，长期留存）"""
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        # 记录【保留工作树】事件
        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"

# 全局单例：工作树管理器
WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


# -- 基础通用工具函数：文件读写、Shell执行、路径安全校验 --
def safe_path(p: str) -> Path:
    """
    路径安全校验：禁止访问工作目录以外的路径（防路径穿越）
    :param p: 相对路径字符串
    :return: 解析后的绝对Path对象
    """
    path = (WORKDIR / p).resolve()
    # 校验路径是否在当前工作目录范围内
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """在主工作目录执行Shell命令，内置高危命令拦截"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容，支持行数截断"""
    try:
        lines = safe_path(path).read_text().splitlines()
        # 限制读取行数
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        # 限制总输出字符数
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件，自动创建上级目录"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """文件精准替换：查找指定原文并替换（仅替换第一个匹配项）"""
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具映射表：工具名称 -> 对应处理函数
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

# 大模型工具定义列表：声明工具名称、描述、入参结构（符合Anthropic Tool Call规范）
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace (blocking).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "task_create",
        "description": "Create a new task on the shared task board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "task_list",
        "description": "List all tasks with status, owner, and worktree binding.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "task_get",
        "description": "Get task details by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "task_update",
        "description": "Update task status or owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                },
                "owner": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "worktree": {"type": "string"},
                "owner": {"type": "string"},
            },
            "required": ["task_id", "worktree"],
        },
    },
    {
        "name": "worktree_create",
        "description": "Create a git worktree and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "integer"},
                "base_ref": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_list",
        "description": "List worktrees tracked in .worktrees/index.json.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "worktree_status",
        "description": "Show git status for one worktree.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_run",
        "description": "Run a shell command in a named worktree directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "worktree_remove",
        "description": "Remove a worktree and optionally mark its bound task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "force": {"type": "boolean"},
                "complete_task": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept in lifecycle state without removing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_events",
        "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
]


def agent_loop(messages: list):
    """
    智能代理主循环：大模型对话 + 工具调用闭环
    :param messages: 历史对话消息列表
    流程：调用大模型 -> 解析工具调用 -> 执行工具 -> 结果回传给大模型，循环直到无需调用工具
    """
    while True:
        # 调用Anthropic大模型，传入对话历史、系统提示、工具列表
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # 保存大模型本轮输出
        messages.append({"role": "assistant", "content": response.content})
        # 停止条件：大模型不再请求调用工具，退出循环
        if response.stop_reason != "tool_use":
            return

        results = []
        # 遍历所有工具调用请求
        for block in response.content:
            if block.type == "tool_use":
                # 根据工具名匹配对应处理函数
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # 执行工具并获取输出
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                # 控制台简略打印工具执行结果
                print(f"> {block.name}:")
                print(str(output)[:200])
                # 组装工具返回结果，回传给大模型
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        # 将工具执行结果作为用户消息，继续下一轮对话
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # 程序入口：交互式命令行终端
    print(f"Repo root for s12: {REPO_ROOT}")
    # 检测Git环境并给出提示
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")

    # 对话历史存储列表
    history = []
    # 交互式输入循环
    while True:
        try:
            # 读取用户输入指令
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # 捕获 Ctrl+C / Ctrl+D，退出程序
            break
        # 空输入 / q / exit 触发退出
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 保存用户指令到对话历史
        history.append({"role": "user", "content": query})
        # 启动代理主循环处理指令
        agent_loop(history)
        # 打印最终回复文本
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()