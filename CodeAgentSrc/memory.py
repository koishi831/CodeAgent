import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, List


MEMORY_PROMPT_TEMPLATE = """# 记忆系统
你在 `{memory_dir}` 目录下拥有一套基于文件的持久化记忆系统。

## 记忆保存方法
使用文件写入工具创建带 YAML 头部信息的记忆文件：

```markdown
---
name: 记忆名称
description:  一句话简述
---
记忆正文内容
```

存放路径: `{memory_dir}/`
文件名命名规则: `{{slugified_name}}.md`

需要删除记忆文件时，你用run_shell命令删除。
写入记忆目录后，MEMORY.md 索引文件会自动更新，请勿手动修改。

## 禁止存入内容
- 代码范式与架构设计（直接读取源码即可）
- 临时性、短期无效的任务细节

## 调取记忆时机
用户明确要求记住 / 调取内容时，或是过往对话上下文存在关联信息时，自动调用对应记忆。
{index_section}"""

def _project_hash() -> str:
    return hashlib.sha256(str(Path.cwd()).encode()).hexdigest()[:16]

def get_memory_dir() -> Path:
    d = Path(__file__).parent / ".codeagent" / _project_hash() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _parse_memory_file(file_path: Path) -> Optional[Dict]:
    """解析带 YAML 头部的记忆文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 匹配 YAML 头部
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not match:
            return None
        
        yaml_content = match.group(1)
        body = match.group(2).strip()
        
        # 解析 YAML 头部（简单解析，不用完整 YAML 库）
        metadata = {}
        for line in yaml_content.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        
        # 确保必需字段存在
        if "name" not in metadata:
            metadata["name"] = file_path.stem
        if "description" not in metadata:
            metadata["description"] = ""
        
        return {
            "file_path": file_path,
            "name": metadata["name"],
            "description": metadata["description"],
            "body": body
        }
    except Exception:
        return None

def _regenerate_index() -> str:
    """重新生成记忆索引内容（扫描所有记忆文件）"""
    memory_dir = get_memory_dir()
    
    memories: List[Dict] = []
    
    # 扫描所有记忆文件
    for file_path in memory_dir.glob("*.md"):
        if file_path.name == "MEMORY.md":
            continue
        parsed = _parse_memory_file(file_path)
        if parsed:
            memories.append(parsed)
    
    # 生成索引文本
    if not memories:
        return "\n## 记忆索引\n(暂无记忆)"
    
    index_text = "\n## 记忆索引\n"
    for mem in memories:
        index_text += f"- [{mem['name']}]({mem['file_path'].name}): {mem['description']}\n"
    
    return index_text


def _generate_index() -> str:
    """获取记忆索引内容（优先读取 MEMORY.md）"""
    memory_dir = get_memory_dir()
    index_file = memory_dir / "MEMORY.md"
    
    # 优先读取已有的 MEMORY.md
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    
    # 如果读取失败或文件不存在，重新生成并保存
    index_content = _regenerate_index()
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(index_content)
    return index_content

def update_memory_index(new_file_path: Optional[Path] = None) -> None:
    """更新 MEMORY.md 索引文件"""
    memory_dir = get_memory_dir()
    index_file = memory_dir / "MEMORY.md"
    index_content = _regenerate_index()
    
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(index_content)

def build_memory_prompt() -> str:
    template = MEMORY_PROMPT_TEMPLATE
    memory_dir = get_memory_dir()
    index_section = _generate_index()
    template = template.replace("{memory_dir}", str(memory_dir))
    template = template.replace("{index_section}", index_section)
    return template
