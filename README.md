# CodeAgent

基于大模型的智能编程助手，支持代码编辑、文件操作、终端命令执行等功能。

## 功能特性

- 🤖 **智能对话**：自然语言交互，理解编程需求
- 📝 **代码编辑**：读取、写入、修改文件
- 🔍 **代码搜索**：grep 搜索、文件列表查看
- 💻 **终端执行**：运行 shell 命令
- 🌐 **网络请求**：web_fetch 获取网页内容
- 🧠 **多Agent协作**：主Agent + 子Agent（explore/plan/general）
- 💾 **记忆系统**：持久化用户偏好和项目信息，主Agent和子Agent共用
- ⚡ **上下文管理**：自动裁剪和压缩对话历史
- 📊 **Token统计**：实时显示费用消耗

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的配置：

```bash
cp .env.example .env
```

编辑 `.env`：
```
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL_ID=deepseek-v4-flash
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行程序

```bash
python start.py
```

或者：
```bash
python -m CodeAgentSrc
```

## 特殊命令

程序运行时可以使用以下命令：

| 命令 | 说明 |
|------|------|
| `/exit` | 退出程序 |
| `/clear` | 清空对话历史 |
| `/cost` | 查看token消耗统计 |
| `/compact` | 手动压缩对话上下文 |
| `/memory` | 打开记忆目录 |

## 目录结构

```
CodeAgent/
├── CodeAgentSrc/
│   ├── __main__.py      # 主程序入口
│   ├── agent.py         # 核心Agent类
│   ├── memory.py        # 记忆系统
│   ├── prompt.py        # 系统提示词
│   ├── subagent.py      # 子Agent系统
│   ├── tool.py          # 工具系统
│   └── ui.py            # 用户界面
├── .env.example         # 环境变量示例
├── .gitignore
├── requirements.txt     # 依赖包
├── start.py             # 启动脚本
└── README.md
```

### 核心模块设计思路

#### 1. agent.py - 核心Agent类

**设计思路**：
- 单一职责：专注于对话管理、工具调用、上下文管理
- 状态管理：使用 `_last_token_count` 和 `_last_message_count` 追踪状态
- 增量估算：利用上次API返回的精确值 + 新增部分估算，提高准确性

**核心功能**：
- 对话历史管理
- 工具调用与执行
- 上下文裁剪与压缩
- 子Agent调用
- Token消耗统计

#### 2. tool.py - 工具系统

**设计思路**：
- 工具定义与执行分离
- 权限检查：危险操作需要用户确认
- 子Agent工具区分：不同类型子Agent有不同工具集

**内置工具**：
- `read_file` - 读取文件
- `write_file` - 写入文件
- `edit_file` - 编辑文件（精确匹配）
- `list_files` - 列出目录
- `grep_search` - 代码搜索
- `run_shell` - 执行终端命令
- `web_fetch` - 获取网页内容
- `agent` - 调用子Agent

#### 3. memory.py - 记忆系统

**设计思路**：
- 文件系统存储：每个记忆一个 Markdown 文件
- 自动索引：更新记忆时自动生成索引
- 增量加载：短记忆直接显示在提示词中，长记忆按需读取
- 元数据支持：使用 YAML 头部存储记忆名称和描述
- **主Agent和子Agent共用**：记忆系统被集成到所有Agent的系统提示词中

**记忆类型**：
- 用户偏好
- 项目规范
- 重要上下文

#### 4. prompt.py - 系统提示词

**设计思路**：
- 模板化：使用占位符动态替换
- 环境感知：包含工作目录、日期、平台等信息
- 模块化：从 memory 和 subagent 加载相关内容

**提示词内容**：
- 系统角色定义
- 工具使用规范
- 子Agent说明
- 记忆信息
- 当前环境信息

#### 5. subagent.py - 子Agent系统

**设计思路**：
- 三种类型：explore（只读检索）、plan（方案规划）、general（通用任务）
- 专用提示词：每种类型有专门优化的系统提示词
- 工具限制：explore/plan 只读，general 完整工具集
- 避免过度设计：强调简单任务用简单方案
- 记忆集成：子Agent也能使用记忆系统

**子Agent类型**：

| 类型    | 工具权限                            | 用途              |
|---------|------------------------------------|------------------|
| explore | read_file, list_files, grep_search | 代码检索、项目探索 |
| plan    | read_file, list_files, grep_search | 方案规划、任务分解 |
| general | 全部工具                            | 独立执行完整任务  |

#### 6. ui.py - 用户界面

**设计思路**：
- 纯输出层：不包含业务逻辑
- Rich 美化：使用 Rich 库提供美观的终端输出
- 费用计算：实时显示 token 消耗和费用

**UI组件**：
- 对话输出
- 工具调用提示
- 费用统计展示
- 用户确认对话框

## 上下文管理策略

### 触发条件

当对话历史 token 超过上下文窗口的 40% 时触发。

### 处理流程

1. 保留系统提示词
2. 保留最近 2 轮对话
3. 将中间历史压缩为摘要
4. 使用摘要 + 最近对话继续

### 优化点

- 增量 token 估算：利用上次 API 精确值
- 智能裁剪：只裁剪必要部分
- 摘要质量：用 LLM 生成高质量摘要

## 子Agent调用原则

### 何时使用子Agent？

- 当前任务与子Agent功能描述匹配时
- 适合执行独立查询
- 避免大量返回内容挤占主上下文窗口

## 开发说明

### 添加新工具

在 `tool.py` 中：
1. 添加工具函数
2. 在 `tool_definitions` 中定义
3. 根据需要添加权限检查

### 添加新的子Agent类型

在 `subagent.py` 中：
1. 添加系统提示词
2. 在 `get_tools_for_agent_type` 中配置工具
3. 在提示词中添加说明
