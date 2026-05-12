# 🔀 LLM Smart Gateway v2

> **同名模型跨渠道合并 × 负载均衡 × auto 智能路由** — one-api / new-api 的增强层

## 解决什么问题？

one-api / new-api 统一了 API key，但同一个底层模型在不同渠道叫不同名字：

| 渠道 | model_id | 实际模型 |
|------|----------|---------|
| DeepSeek 官方 | `deepseek-chat` | DeepSeek V4 |
| 代理商A | `deepseek-v4-pro` | DeepSeek V4 |
| 代理商B | `deepseek-ai/deepseek-chat` | DeepSeek V4 |
| OpenRouter | `deepseek/deepseek-chat` | DeepSeek V4 |

**4 个名字 = 同一个模型，但 one-api 把它们当 4 个独立模型，无法负载均衡。**

Smart Gateway v2 通过**模型指纹识别**，把这 4 个名字合并为逻辑模型 `deepseek-v4`，再在 4 个渠道间智能均衡。

## 核心能力

### 1. 模型指纹识别 — 同名模型自动合并

内置 37 种逻辑模型的指纹库，通过别名+正则匹配识别：

```
deepseek-chat          ─┐
deepseek-v4-pro         │  → deepseek-v4 (逻辑模型)
deepseek-ai/deepseek-chat │
deepseek/deepseek-chat ─┘
```

指纹表可自定义导出/导入（JSON），支持添加私有模型别名。

### 2. 跨渠道负载均衡

同一逻辑模型下的多个渠道实例，6 种均衡策略：

| 策略 | 逻辑 | 适用场景 |
|------|------|---------|
| `weighted` | 综合权重(静态×成功率×延迟) | 默认，生产推荐 |
| `round_robin` | 轮询 | 均匀分布 |
| `least_loaded` | 最久未用优先 | 冷启动均衡 |
| `fastest` | 最低延迟优先 | 实时性要求高 |
| `priority` | 渠道优先级 | 官方优先 |
| `failover` | 优先级+故障转移 | 高可用 |

**熔断机制**：连续 3 次失败自动禁用渠道 30 秒，恢复后逐步恢复流量。

### 3. auto 智能路由

`model="auto"` 自动分类任务→选最佳逻辑模型→跨渠道均衡：

```
"写一个排序算法"   → code     → qwen3-coder / 渠道均衡
"分析市场趋势"     → reasoning → deepseek-r1 / 渠道均衡
"这张图片是什么"   → vision   → gemini-2.5-pro
"你好"            → chat     → glm-5
```

6 种任务类型：`code` / `reasoning` / `vision` / `chat` / `safety` / `embedding`

### 4. one-api 集成

直接从 one-api 数据库/API 拉取渠道配置，自动注册+合并：

```python
from oneapi_adapter import OneAPIAdapter
adapter = OneAPIAdapter(gateway)
result = adapter.load_from_api("http://localhost:3000", admin_token)
# 或从导出的 JSON:
result = adapter.load_from_json("channels.json")
```

## 架构

```
┌──────────┐  model="auto"   ┌───────────────────────────────────────────┐
│  Client   │────────────────▶│           SmartGateway                    │
│ (openai)  │                 │                                          │
└──────────┘                 │  ┌─────────┐    ┌──────────────────────┐  │
                             │  │ Fingerprint│  │  ModelRegistry      │  │
                             │  │    DB     │  │                      │  │
                             │  │          │  │ deepseek-v4:         │  │
                             │  │ deepseek  │  │  ├ 渠道A: ds-chat    │──┼──▶ 渠道A API
                             │  │  -chat ──│─▶│  ├ 渠道B: ds-v4-pro  │──┼──▶ 渠道B API
                             │  │  -v4-pro─│─▶│  ├ 渠道C: ds/chat    │──┼──▶ 渠道C API
                             │  │  -/chat ─│─▶│  └ 渠道D: ds/chat    │──┼──▶ 渠道D API
                             │  └─────────┘    └──────────────────────┘  │
                             │                        │                  │
                             │                 ┌──────▼──────┐          │
                             │                 │ LoadBalancer │          │
                             │                 │ weighted/rr/ │          │
                             │                 │ failover...  │          │
                             │                 └─────────────┘          │
                             └───────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
git clone https://github.com/sumo1111/llm-smart-gateway.git
cd llm-smart-gateway
pip install httpx  # 可选
```

### Python 使用

```python
from gateway import SmartGateway
from balancer import BalanceStrategy
from model_fingerprint import ModelTier

gw = SmartGateway(balance_strategy=BalanceStrategy.WEIGHTED)

# 添加渠道
gw.add_channel(
    channel_id="deepseek-official",
    channel_name="DeepSeek 官方",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-xxx",
    model_ids=["deepseek-chat", "deepseek-reasoner", "deepseek-coder"],
    priority=10,
    weight=2,
)

gw.add_channel(
    channel_id="reseller-a",
    channel_name="代理商A",
    base_url="https://reseller.example.com/v1",
    api_key="sk-yyy",
    model_ids=["deepseek-v4-pro", "glm-5.1"],
    priority=5,
    weight=1,
)

# 查看合并结果
ds = gw.registry.get("deepseek-v4")
print(f"deepseek-v4: {len(ds.instances)} 个渠道实例")

# auto 路由
result = gw.resolve_model("auto", prompt="写一个排序算法")
# → {"canonical": "qwen3-coder", "model_id": "...", "channel_id": "...", ...}

# 逻辑模型名路由 (跨渠道均衡)
result = gw.resolve_model("deepseek-v4", prompt="test")
# → 自动在 4 个渠道间均衡选择

# 原始 ID 路由 (自动指纹识别)
result = gw.resolve_model("deepseek-chat", prompt="test")
# → 自动识别为 deepseek-v4, 然后跨渠道均衡

# 完整聊天调用
response = gw.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="auto",  # 或 "deepseek-v4" 或 "deepseek-chat"
)
```

### one-api 集成

```python
from gateway import SmartGateway
from oneapi_adapter import OneAPIAdapter

gw = SmartGateway()
adapter = OneAPIAdapter(gw)

# 方式1: 从 one-api API 拉取
result = adapter.load_from_api(
    base_url="http://localhost:3000",
    admin_token="your-admin-token",
)

# 方式2: 从导出的 JSON 加载
result = adapter.load_from_json("channels_export.json")

# 方式3: 从环境变量
# ONE_API_BASE_URL=http://localhost:3000
# ONE_API_ADMIN_TOKEN=xxx
result = adapter.load_from_env()
```

### 指纹库管理

```bash
# 列出内置指纹
python model_fingerprint.py list

# 识别模型 ID
python model_fingerprint.py identify deepseek-chat glm-5.1 unknown-model

# 导出指纹库 (方便自定义)
python model_fingerprint.py export -o my_fingerprints.json
```

## 测试结果

```
Phase 1: 模型指纹识别 ...................... ✅ (37 种逻辑模型)
Phase 2: 逻辑模型注册表 (同名合并) ........... ✅ (4渠道 deepseek-chat → 1个逻辑端点)
Phase 3: 跨渠道负载均衡 .................... ✅ (6种策略)
Phase 3+: 熔断与恢复 ...................... ✅ (连续3次失败→熔断30s→自动恢复)
Phase 4+: 任务分类器 ...................... ✅ (7/7 正确分类)
Phase 4: auto 智能路由 .................... ✅ (auto/逻辑名/原始ID 三种模式)
Phase 5: one-api 适配器 ................... ✅ (JSON/API/ENV 三种加载)
全链路: 注册→合并→路由→均衡 ............... ✅ (含故障转移)
```

## 文件结构

```
llm-smart-gateway/
├── model_fingerprint.py  # Phase 1: 模型指纹识别
├── registry.py           # Phase 2: 逻辑模型注册表
├── balancer.py           # Phase 3: 跨渠道负载均衡器
├── gateway.py            # Phase 4: auto 智能网关
├── oneapi_adapter.py     # Phase 5: one-api 适配器
├── test_all.py           # 全链路测试
└── README.md
```

## 与 one-api 的关系

**Smart Gateway 是 one-api 的增强层，不是替代品。**

```
Client → Smart Gateway (模型合并+均衡+auto) → one-api (统一key+计费+管理) → 上游API
```

也可以绕过 one-api 直接对接上游 API：

```
Client → Smart Gateway → 多个上游 API
```

## License

MIT
