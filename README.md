# LangGraph 智能体演示 — AWS Bedrock 应用推理配置文件

基于 **LangChain** + **LangGraph** 构建的 ReAct 风格智能体演示，使用 AWS Bedrock **应用推理配置文件（AIP）** 作为 LLM 后端。

## 架构

```
START → agent ──(有工具调用?)──► tools ──► agent (循环)
                └──(无工具调用)──► END
```

智能体遵循 **思考 → 行动 → 观察** 循环：
1. LLM 对查询进行推理并决定调用哪个工具
2. 工具节点执行所请求的工具
3. 结果反馈给 LLM 进行下一步推理
4. 循环持续直到 LLM 给出最终答案

## 文件说明

| 文件 | 描述 |
|------|------|
| `agent.py` | 主智能体图、LLM 配置、交互式 CLI |
| `tools.py` | 自定义 LangChain 工具 |
| `visualize_graph.py` | ASCII 及可选 PNG 图形可视化 |
| `requirements.txt` | Python 依赖 |

## 环境准备

### 1. 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 AWS 凭证

本演示使用默认 AWS 凭证链，支持以下方式之一：

```bash
# 方式 A：环境变量
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...   # 使用临时凭证时需要

# 方式 B：AWS Profile
export AWS_PROFILE=my-profile

# 方式 C：IAM 角色（EC2/Lambda/ECS 上自动生效）
```

Bedrock 客户端区域设置为 `ap-northeast-1`（东京），与 AIP 区域匹配。

## 使用方法

### 交互模式（多轮对话）

```bash
source .venv/bin/activate
python agent.py
```

### 自动演示查询

```bash
python agent.py --demo
```

### 可视化图结构

```bash
python visualize_graph.py
```

## 应用推理配置文件

| 配置项 | 值 |
|--------|-----|
| ARN | `arn:aws:bedrock:ap-northeast-1:123456789012:application-inference-profile/qs2x0oirm4py` |
| 区域 | `ap-northeast-1` |

> **注意**：请将 ARN 中的 `123456789012` 替换为您自己的 AWS 账号 ID。

AIP ARN 直接作为 `model=AIP_ARN` 传入 `ChatBedrockConverse`，由其负责路由到底层基础模型。

## 可用工具

| 工具 | 描述 |
|------|------|
| `calculator` | 计算数学表达式（支持 `sqrt`、`log`、`pi` 等 `math` 函数） |
| `get_current_datetime` | 获取当前 UTC 或本地日期时间 |
| `get_weather` | 主要城市的模拟天气数据 |
| `search_knowledge_base` | 包含 LangChain、LangGraph、Bedrock 等信息的模拟知识库 |
| `unit_converter` | 单位转换：km↔miles、kg↔lbs、摄氏度↔华氏度等 |

## 示例查询

```
sqrt(256) + log(100) 等于多少？
东京和巴黎的天气如何？哪个更暖？
将 100 英里转换为千米，30°C 转换为华氏度。
LangGraph 是什么？它与 LangChain 有何不同？
现在 UTC 时间是几点？
```
