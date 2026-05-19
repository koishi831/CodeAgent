"""
subagent.py - 子 Agent 系统模块
==========================

本模块定义了可用的子 Agent 类型及其系统提示词，让主 Agent 可以调用
专业化的子 Agent 完成特定任务。

子 Agent 类型：
- explore: 代码探查（只读），用于全面浏览和分析代码库
- plan: 方案规划（只读），用于制定结构化实施方案
- general: 通用任务（全工具），用于独立完成任务

每个子 Agent 都有自己独立的上下文和工具限制。
"""
import platform
import os
from datetime import date
from pathlib import Path
from .memory import build_memory_prompt


EXPLORE_PROMPT = """你是文件检索助手，擅长全面浏览、探查各类代码项目仓库。

重要限定：只读模式 —— 禁止任何文件修改
本次仅执行只读探查任务，严格禁止以下所有操作：
- 新建文件（禁止写入文件、创建空文件及一切文件创建行为）
- 修改已有文件（禁止执行编辑文件操作）
- 删除文件（禁止使用删除命令及任何删除行为）
- 运行任何会改变系统状态的指令

你的工作仅限检索与分析现有代码。

核心能力：
- 使用 glob 模式匹配文件
- 使用 regex 搜索文件内容
- 分析文件内容

执行规范：
- 使用 list_files 匹配文件模式
- 使用 grep_search 搜索文件内容
- 使用 read_file 读取具体文件内容
- 根据调用者指定的详细程度调整搜索方法

合理利用可用工具高效完成用户检索需求，并清晰明了地汇总输出探查结果。

# 环境信息
工作目录: {{cwd}}
日期: {{date}}
平台: {{platform}}
终端 Shell: {{shell}}
{{memory}}
"""

PLAN_PROMPT = """你是方案规划助手，一款只读专用子代理，负责设计实施方案。

重要限制：
- 仅能使用 read_file, list_files, grep_search 工具。
- 仅为只读权限，不能修改任何文件。

你的任务：
- 分析代码库以理解当前架构，新项目没有这一步
- 制定分步骤落地执行方案
- 确定需要改动的核心文件
- 权衡架构设计中的取舍与利弊

输出结构化实施方案，包含：
1. 项目现状总结，新项目没有这一步
2. 分步骤实现步骤
3. 实现所需核心文件
4. 潜在风险与注意事项

# 环境信息
工作目录: {{cwd}}
日期: {{date}}
平台: {{platform}}
终端 Shell: {{shell}}
{{memory}}
"""

GENERAL_PROMPT = """你是通用任务助手，依据用户指令调用可用工具完成任务。
务必完整执行任务，不画蛇添足，也不半途而废。
任务完成后，输出精简工作报告，写明执行内容与核心结论即可，仅保留关键信息，由调用方转达给用户。

核心能力：
- 搜索大型代码库中的代码、配置和模式
- 分析多个文件以理解系统架构
- 研究复杂问题，需要探索多个文件的内容
- 执行多步骤研究任务

执行规范：
- 为文件搜索指定范围，当不知道文件位置时使用 glob 模式
- 当知道文件路径时使用 read_file 读取文件内容
- 分析文件内容，提取关键信息
- 仅在必要时创建新文件，否则优先编辑已存在文件。

# 环境信息
工作目录: {{cwd}}
日期: {{date}}
平台: {{platform}}
终端 Shell: {{shell}}
{{memory}}
"""


def get_available_agent_types() -> list[dict[str, str]]:
    """
    获取所有可用的子 Agent 类型

    返回:
        字典列表，每个字典包含 name 和 description 字段
    """
    types = [
        {"name": "explore", "description": "Fast, read-only codebase search and exploration"},
        {"name": "plan", "description": "Read-only analysis with structured implementation plans"},
        {"name": "general", "description": "Full tools for independent tasks"},
    ]
    return types


def build_agent_descriptions() -> str:
    """
    构建用于系统提示词的子 Agent 描述

    将所有子 Agent 类型转换为 Markdown 格式的列表，
    用于插入到主系统提示词中。

    返回:
        Markdown 格式的子 Agent 说明列表
    """
    types = get_available_agent_types()
    descriptions = []
    for agent_type in types:
        descriptions.append(f"- **{agent_type['name']}**: {agent_type['description']}")
    return "\n".join(descriptions)


def get_prompt_for_agent_type(agent_type: str) -> str:
    """
    根据 agent 类型获取对应的原始提示词模板（未替换占位符）

    参数:
        agent_type: 子 Agent 类型（"explore" | "plan" | "general"）

    返回:
        对应类型的原始提示词模板字符串，未知类型返回通用提示词
    """
    if agent_type == "explore":
        return EXPLORE_PROMPT
    elif agent_type == "plan":
        return PLAN_PROMPT
    elif agent_type == "general":
        return GENERAL_PROMPT
    else:
        return GENERAL_PROMPT


def build_subagent_system_prompt(agent_type: str) -> str:
    """
    构建子 Agent 的完整系统提示词（替换环境信息占位符）

    参数:
        agent_type: 子 Agent 类型（"explore" | "plan" | "general"）

    返回:
        完整的系统提示词字符串
    """
    template = get_prompt_for_agent_type(agent_type)
    template = template.replace("{{cwd}}", str(Path.cwd()))
    template = template.replace("{{date}}", str(date.today().isoformat()))
    template = template.replace("{{platform}}", f"{platform.system()} {platform.machine()}")
    template = template.replace("{{shell}}", os.environ.get("ComSpec") or "")
    template = template.replace("{{memory}}", build_memory_prompt())
    return template
