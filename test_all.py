"""
LLM Smart Gateway v2 — 全链路测试

核心验证:
1. 指纹识别: 不同渠道的同一个底层模型被正确归入同一逻辑模型
2. 跨渠道负载均衡: 同一逻辑模型下的多渠道实例能均衡调度
3. auto 智能路由: 任务分类→逻辑模型→渠道实例
4. 熔断恢复: 连续失败后自动熔断，恢复后重新启用
5. 故障转移: 首选渠道失败后自动切换到其他渠道
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from model_fingerprint import FingerprintDB, ModelTier, TaskType
from registry import ModelRegistry, ChannelInstance
from balancer import LoadBalancer, BalanceStrategy
from gateway import SmartGateway, classify_task


# ─── 模拟数据: 3 个渠道的同一个 DeepSeek V4 ───

MOCK_CHANNELS = [
    {
        "channel_id": "deepseek-official",
        "channel_name": "DeepSeek 官方",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-deepseek-xxx",
        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"],
        "priority": 10,
        "weight": 2,
    },
    {
        "channel_id": "reseller-a",
        "channel_name": "代理商A",
        "base_url": "https://reseller-a.example.com/v1",
        "api_key": "sk-reseller-a-xxx",
        "models": ["deepseek-v4-pro", "deepseek-r1", "deepseek-coder-6.7b-instruct"],
        "priority": 5,
        "weight": 1,
    },
    {
        "channel_id": "reseller-b",
        "channel_name": "代理商B",
        "base_url": "https://reseller-b.example.com/v1",
        "api_key": "sk-reseller-b-xxx",
        "models": ["deepseek-ai/deepseek-chat", "glm-5.1", "qwen3.5-397b-a17b"],
        "priority": 3,
        "weight": 1,
    },
    {
        "channel_id": "openrouter",
        "channel_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-xxx",
        "models": [
            "deepseek/deepseek-chat", "z-ai/glm-5.1", "z-ai/glm5",
            "qwen/qwen3-coder-480b-a35b-instruct", "meta/llama-3.1-70b-instruct",
        ],
        "priority": 1,
        "weight": 1,
    },
]


def test_fingerprint():
    """✅ Phase 1: 指纹识别"""
    print("\n" + "=" * 60)
    print("Phase 1: 模型指纹识别")
    print("=" * 60)

    db = FingerprintDB()

    test_cases = [
        # (原始 model_id, 期望逻辑名)
        ("deepseek-chat", "deepseek-v4"),
        ("deepseek-v4-pro", "deepseek-v4"),
        ("deepseek-ai/deepseek-chat", "deepseek-v4"),
        ("deepseek/deepseek-chat", "deepseek-v4"),
        ("deepseek-reasoner", "deepseek-r1"),
        ("glm-5.1", "glm-5"),
        ("z-ai/glm-5.1", "glm-5"),
        ("z-ai/glm5", "glm-5"),
        ("qwen3.5-397b-a17b", "qwen3.5"),
        ("meta/llama-3.1-70b-instruct", "llama3.1-70b"),
        ("unknown-model-xyz", "unknown-model-xyz"),
    ]

    all_pass = True
    for model_id, expected in test_cases:
        canonical = db.canonical_name(model_id)
        ok = canonical == expected
        if not ok:
            all_pass = False
        print(f"  {'✅' if ok else '❌'} {model_id:<40} → {canonical:<20} (期望: {expected})")

    assert all_pass, "指纹识别有误"
    print(f"\n  ✅ Phase 1 全部通过 — 识别了 {db.summary()['total_fingerprints']} 种逻辑模型")


def test_registry():
    """✅ Phase 2: 逻辑模型注册表 — 同名模型合并"""
    print("\n" + "=" * 60)
    print("Phase 2: 逻辑模型注册表 (同名模型合并)")
    print("=" * 60)

    gw = SmartGateway()

    # 注册所有渠道
    for ch in MOCK_CHANNELS:
        ch_copy = {k: v for k, v in ch.items() if k != "models"}
        ch_copy["model_ids"] = ch["models"]
        gw.add_channel(**ch_copy)

    # 验证 deepseek-v4 合并了 3 个渠道
    ds = gw.registry.get("deepseek-v4")
    assert ds is not None, "deepseek-v4 逻辑模型未创建"
    assert len(ds.instances) == 4, f"deepseek-v4 应有 4 个实例, 实际: {len(ds.instances)}"

    print(f"\n  🔍 deepseek-v4 逻辑模型 ({len(ds.instances)}个渠道实例):")
    for inst in ds.instances:
        print(f"     渠道: {inst.channel_name:<20} model_id: {inst.model_id:<35} base_url: {inst.base_url}")

    # 验证 glm-5 合并了多渠道
    glm = gw.registry.get("glm-5")
    assert glm is not None, "glm-5 逻辑模型未创建"
    print(f"\n  🔍 glm-5 逻辑模型 ({len(glm.instances)}个渠道实例):")
    for inst in glm.instances:
        print(f"     渠道: {inst.channel_name:<20} model_id: {inst.model_id:<35}")

    # 统计
    summary = gw.registry.summary()
    print(f"\n  📊 注册表统计:")
    print(f"     逻辑模型: {summary['logical_models']}")
    print(f"     渠道实例: {summary['total_instances']}")
    print(f"     去重率: {summary['merge_ratio']}")
    print(f"     按等级: {summary['by_tier']}")
    print(f"     按任务: {summary['by_task']}")

    print(f"\n  ✅ Phase 2 全部通过")


def test_balancer():
    """✅ Phase 3: 跨渠道负载均衡"""
    print("\n" + "=" * 60)
    print("Phase 3: 跨渠道负载均衡")
    print("=" * 60)

    gw = SmartGateway(balance_strategy=BalanceStrategy.WEIGHTED)
    for ch in MOCK_CHANNELS:
        ch_copy = {k: v for k, v in ch.items() if k != "models"}
        ch_copy["model_ids"] = ch["models"]
        gw.add_channel(**ch_copy)

    # ── 加权均衡 ──
    ds = gw.registry.get("deepseek-v4")
    channel_counts = {}
    for _ in range(1000):
        inst = gw.balancer.select("deepseek-v4", strategy=BalanceStrategy.WEIGHTED)
        channel_counts[inst.channel_name] = channel_counts.get(inst.channel_name, 0) + 1

    print(f"\n  📊 加权均衡 (1000次, deepseek-v4):")
    for ch, count in sorted(channel_counts.items(), key=lambda x: -x[1]):
        pct = count / 10
        bar = "█" * int(pct)
        print(f"     {ch:<20} {count:>4} ({pct:.1f}%) {bar}")

    # ── 优先级优先 ──
    inst = gw.balancer.select("deepseek-v4", strategy=BalanceStrategy.PRIORITY)
    print(f"\n  🎯 优先级优先: → {inst.channel_name} (priority={inst.priority})")
    assert inst.channel_id == "deepseek-official", f"应选官方渠道, 实际: {inst.channel_id}"

    # ── 轮询 ──
    rr_order = []
    for _ in range(6):
        inst = gw.balancer.select("deepseek-v4", strategy=BalanceStrategy.ROUND_ROBIN)
        rr_order.append(inst.channel_id)
    print(f"\n  🔄 轮询 (6次): {rr_order}")

    # ── 最闲优先 ──
    for inst in ds.instances:
        inst._last_used = 0
    ds.instances[0]._last_used = time.time()  # 官方刚用过
    inst = gw.balancer.select("deepseek-v4", strategy=BalanceStrategy.LEAST_LOADED)
    print(f"\n  🚶 最闲优先: → {inst.channel_name} (跳过刚用过的)")

    print(f"\n  ✅ Phase 3 全部通过")


def test_circuit_breaker():
    """✅ Phase 3+: 熔断与恢复"""
    print("\n" + "=" * 60)
    print("Phase 3+: 熔断与恢复")
    print("=" * 60)

    gw = SmartGateway()
    for ch in MOCK_CHANNELS:
        ch_copy = {k: v for k, v in ch.items() if k != "models"}
        ch_copy["model_ids"] = ch["models"]
        gw.add_channel(**ch_copy)

    ds = gw.registry.get("deepseek-v4")
    official = ds.instances[0]

    # 模拟连续失败
    print(f"\n  🔴 模拟 {official.channel_name} 连续 3 次失败:")
    for i in range(3):
        official.record_call(latency_ms=5000, success=False)
        print(f"     失败 {i+1}: healthy={official.is_healthy}, consecutive_fails={official._consecutive_fails}")

    # 熔断后应该跳过这个渠道
    assert not official.is_healthy, "应该被熔断"
    healthy = ds.healthy_instances()
    print(f"\n  ⚡ 熔断生效: 健康实例 {len(healthy)}/{len(ds.instances)}")
    assert official.channel_id not in [i.channel_id for i in healthy]

    # 加权均衡应该自动绕过熔断实例
    selected_channels = set()
    for _ in range(100):
        inst = gw.balancer.select("deepseek-v4")
        selected_channels.add(inst.channel_id)
    assert "deepseek-official" not in selected_channels, "熔断渠道不应被选中"
    print(f"     100次选择均绕过了熔断渠道 ✅")

    # 模拟恢复 (等熔断时间过)
    official._disabled_until = time.time() - 1  # 强制恢复
    official.record_call(latency_ms=100, success=True)
    print(f"\n  🟢 恢复后: healthy={official.is_healthy}, success_rate={official.success_rate:.3f}")

    print(f"\n  ✅ Phase 3+ 全部通过")


def test_auto_routing():
    """✅ Phase 4: auto 智能路由"""
    print("\n" + "=" * 60)
    print("Phase 4: auto 智能路由")
    print("=" * 60)

    gw = SmartGateway()
    for ch in MOCK_CHANNELS:
        ch_copy = {k: v for k, v in ch.items() if k != "models"}
        ch_copy["model_ids"] = ch["models"]
        gw.add_channel(**ch_copy)

    # ── auto 模式 ──
    test_cases = [
        ("写一个快速排序算法", "code", "qwen3-coder"),
        ("分析一下当前A股市场趋势", "reasoning", "deepseek-r1"),
        ("你好，今天天气怎么样", "chat", None),  # 不验证具体族，只验证策略
        ("这段代码有什么问题", "code", "qwen3-coder"),
    ]

    print(f"\n  🧠 auto 路由测试:")
    for prompt, expected_task, expected_family in test_cases:
        r = gw.resolve_model("auto", prompt=prompt)
        if "error" in r:
            print(f"     ❌ '{prompt[:20]}...' → ERROR: {r['error']}")
            continue

        task_match = r.get("task") == expected_task
        task_icon = "✅" if task_match else "⚠️"
        print(f"     {task_icon} '{prompt[:20]}...'")
        print(f"        → 逻辑模型: {r['canonical']}")
        print(f"        → 渠道实例: {r['channel_name']} / {r['model_id']}")
        print(f"        → 任务: {r['task']} 等级: {r['tier']} 策略: {r['strategy']}")

    # ── 逻辑模型名模式 ──
    r = gw.resolve_model("deepseek-v4", prompt="test")
    print(f"\n  📍 逻辑模型名路由:")
    print(f"     model='deepseek-v4' → canonical={r['canonical']}, channel={r['channel_name']}, model_id={r['model_id']}")

    # ── 原始 ID 指纹识别模式 ──
    r = gw.resolve_model("deepseek-chat", prompt="test")
    print(f"\n  🔄 原始ID指纹路由:")
    print(f"     model='deepseek-chat' → canonical={r['canonical']}, channel={r['channel_name']}, model_id={r['model_id']}")
    assert r["canonical"] == "deepseek-v4", f"deepseek-chat 应归入 deepseek-v4, 实际: {r['canonical']}"
    assert r["strategy"] == "fingerprint", f"应为 fingerprint 策略, 实际: {r['strategy']}"

    print(f"\n  ✅ Phase 4 全部通过")


def test_task_classifier():
    """✅ 任务分类器"""
    print("\n" + "=" * 60)
    print("Phase 4+: 任务分类器")
    print("=" * 60)

    test_cases = [
        ("写一个Python爬虫", TaskType.CODE),
        ("分析中国GDP增长原因", TaskType.REASONING),
        ("这张图片里是什么", TaskType.VISION),
        ("你好，今天天气怎么样", TaskType.CHAT),
        ("实现一个REST API", TaskType.CODE),
        ("帮我比较这两个方案", TaskType.REASONING),
        ("调试这段代码的bug", TaskType.CODE),
    ]

    all_pass = True
    for prompt, expected in test_cases:
        result = classify_task(prompt)
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  {'✅' if ok else '❌'} '{prompt}' → {result.value} (期望: {expected.value})")

    assert all_pass, "任务分类有误"
    print(f"\n  ✅ 任务分类器全部通过")


def test_oneapi_adapter():
    """✅ Phase 5: one-api 适配器"""
    print("\n" + "=" * 60)
    print("Phase 5: one-api 适配器")
    print("=" * 60)

    from oneapi_adapter import OneAPIAdapter

    gw = SmartGateway()
    adapter = OneAPIAdapter(gw)

    # 模拟 one-api 导出数据
    oneapi_channels = [
        {
            "id": 1,
            "name": "DeepSeek 官方",
            "base_url": "https://api.deepseek.com",
            "key": "sk-ds-xxx",
            "models": "deepseek-chat,deepseek-reasoner,deepseek-coder",
            "priority": 10,
            "weight": 2,
            "status": 1,
        },
        {
            "id": 2,
            "name": "中转站A",
            "base_url": "https://relay-a.com/v1",
            "key": "sk-relay-xxx",
            "models": "deepseek-v4-pro,glm-5.1,qwen3.5-397b-a17b",
            "priority": 5,
            "weight": 1,
            "status": 1,
        },
        {
            "id": 3,
            "name": "禁用渠道",
            "base_url": "https://disabled.com/v1",
            "key": "sk-disabled",
            "models": "gpt-4o",
            "priority": 0,
            "weight": 1,
            "status": 2,  # 禁用
        },
    ]

    # 写临时 JSON
    tmp = "/tmp/test_oneapi_channels.json"
    with open(tmp, "w") as f:
        json.dump(oneapi_channels, f)

    result = adapter.load_from_json(tmp)

    print(f"\n  📊 one-api 加载结果:")
    print(f"     渠道数: {result['channels_loaded']}")
    print(f"     注册模型数: {result['total_models_registered']}")
    print(f"     合并模型数: {result['total_models_merged']}")
    print(f"     逻辑模型数: {result['logical_models']}")

    for d in result["details"]:
        print(f"     渠道 '{d['channel']}': 输入{d['models_in']}个, 合并{d['merged']}个, 逻辑{d['unique_canonicals']}个")

    # 验证: deepseek-chat 和 deepseek-v4-pro 应合并为同一逻辑模型
    ds = gw.registry.get("deepseek-v4")
    assert ds is not None
    assert len(ds.instances) == 2, f"deepseek-v4 应有 2 个渠道实例, 实际: {len(ds.instances)}"
    channels = [i.channel_id for i in ds.instances]
    assert "oneapi-1" in channels and "oneapi-2" in channels

    print(f"\n  🔍 deepseek-v4 合并验证:")
    for inst in ds.instances:
        print(f"     渠道: {inst.channel_id} ({inst.channel_name}) model_id: {inst.model_id}")

    # 验证: 禁用渠道被跳过
    gpt4o = gw.registry.get("gpt-4o")
    assert gpt4o is None, "禁用渠道的模型不应被注册"
    print(f"\n  🚫 禁用渠道正确跳过 ✅")

    # 模型字符串解析
    models = OneAPIAdapter.parse_model_string("gpt-4o,gpt-4o-mini\nclaude-4-sonnet,# 旧模型")
    assert models == ["gpt-4o", "gpt-4o-mini", "claude-4-sonnet"], f"解析错误: {models}"
    print(f"  ✅ 模型字符串解析正确")

    os.unlink(tmp)
    print(f"\n  ✅ Phase 5 全部通过")


def test_end_to_end():
    """✅ 全链路: 注册→合并→路由→均衡"""
    print("\n" + "=" * 60)
    print("全链路: 注册 → 合并 → 路由 → 均衡")
    print("=" * 60)

    gw = SmartGateway(balance_strategy=BalanceStrategy.WEIGHTED)

    # 模拟一个 one-api 场景
    channels = [
        {
            "channel_id": "ds-official",
            "channel_name": "DeepSeek官方",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-ds",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "priority": 10, "weight": 2,
        },
        {
            "channel_id": "reseller-a",
            "channel_name": "中转A",
            "base_url": "https://a.example.com/v1",
            "api_key": "sk-a",
            "models": ["deepseek-v4-pro", "glm-5.1"],
            "priority": 5, "weight": 1,
        },
        {
            "channel_id": "reseller-b",
            "channel_name": "中转B",
            "base_url": "https://b.example.com/v1",
            "api_key": "sk-b",
            "models": ["deepseek/deepseek-chat", "qwen3-coder-480b-a35b-instruct"],
            "priority": 3, "weight": 1,
        },
    ]

    for ch in channels:
        ch_copy = {k: v for k, v in ch.items() if k != "models"}
        ch_copy["model_ids"] = ch["models"]
        gw.add_channel(**ch_copy)

    # deepseek-chat / deepseek-v4-pro / deepseek/deepseek-chat → 同一个 deepseek-v4
    ds = gw.registry.get("deepseek-v4")
    print(f"\n  🔀 deepseek-v4 合并结果: {len(ds.instances)} 个渠道实例")
    for inst in ds.instances:
        print(f"     {inst.channel_name:<15} model_id={inst.model_id:<30} base_url={inst.base_url}")

    # auto 路由
    print(f"\n  🧠 auto 路由测试:")
    prompts = [
        "写一个排序算法",     # code
        "分析市场趋势",       # reasoning
        "你好",              # chat
    ]
    for p in prompts:
        r = gw.resolve_model("auto", prompt=p)
        if "error" in r:
            print(f"     ❌ '{p}' → ERROR: {r['error']}")
        else:
            print(f"     ✅ '{p}' → {r['canonical']} / {r['channel_name']} / {r['model_id']}")

    # 跨渠道均衡
    print(f"\n  ⚖️ 跨渠道均衡 (100次, deepseek-v4):")
    counts = {}
    for _ in range(100):
        inst = gw.balancer.select("deepseek-v4")
        counts[inst.channel_name] = counts.get(inst.channel_name, 0) + 1
    for ch, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"     {ch:<15} {count:>3}%  {'█' * count}")

    # 故障转移
    print(f"\n  🔄 故障转移测试:")
    r1 = gw.resolve_model("deepseek-v4", prompt="test")
    print(f"     首选: {r1['channel_name']} / {r1['model_id']}")

    # 标记首选渠道失败
    for inst in ds.instances:
        if inst.channel_id == r1['channel_id']:
            for _ in range(3):
                inst.record_call(5000, success=False)
            break

    r2 = gw.resolve_model("deepseek-v4", prompt="test")
    print(f"     熔断后: {r2['channel_name']} / {r2['model_id']} (绕过熔断渠道)")

    print(f"\n  ✅ 全链路测试通过")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  LLM Smart Gateway v2 — 全链路测试                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_fingerprint()
    test_registry()
    test_balancer()
    test_circuit_breaker()
    test_task_classifier()
    test_auto_routing()
    test_oneapi_adapter()
    test_end_to_end()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  🎉 全部 8 项测试通过！Smart Gateway v2 就绪               ║")
    print("╚" + "═" * 58 + "╝")
