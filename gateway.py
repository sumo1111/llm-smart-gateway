"""
Phase 4: auto 智能网关 — Smart Gateway

OpenAI 兼容的智能网关代理，核心入口。
支持 model="auto" 智能分流，支持指定逻辑模型名跨渠道均衡。

架构:
  Client → POST /v1/chat/completions (model="auto")
         → SmartGateway._resolve_model()
           1. model="auto"     → 任务分类 → 最佳逻辑模型
           2. model="deepseek-v4" → 逻辑模型 → 跨渠道均衡
           3. model="deepseek-chat" → 指纹识别 → 归入逻辑模型 → 均衡
         → LoadBalancer.select() → 具体渠道实例
         → 转发请求到渠道 → 记录指标 → 返回结果
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Optional

from model_fingerprint import FingerprintDB, ModelTier, TaskType
from registry import ModelRegistry, ChannelInstance, LogicalModel
from balancer import LoadBalancer, BalanceStrategy

logger = logging.getLogger("smart-gateway")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_HTTPX = False


# ─── 任务分类器 ──────────────────────────────────────────────────────────

TASK_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE: [
        "代码", "编程", "函数", "排序", "算法", "debug", "实现", "开发",
        "code", "program", "function", "sort", "algorithm", "implement",
        "python", "javascript", "typescript", "rust", "go", "java",
        "api", "sql", "class", "module", "refactor", "test", "写个",
        "写一个", "帮我写", "编码", "脚本",
    ],
    TaskType.REASONING: [
        "分析", "推理", "为什么", "原因", "证明", "逻辑", "论证",
        "analyze", "reason", "why", "cause", "prove", "logic", "argue",
        "比较", "对比", "评估", "决策", "strategy", "compare",
        "深度", "思考", "研究", "调研",
    ],
    TaskType.VISION: [
        "图片", "截图", "看", "识别", "图像", "视觉", "照片",
        "image", "screenshot", "look", "recognize", "vision", "photo",
        "ocr", "chart", "diagram", "这是什么图",
    ],
    TaskType.CHAT: [
        "你好", "嗨", "hello", "hi", "聊天", "闲聊", "天气",
    ],
    TaskType.SAFETY: [
        "安全", "审核", "过滤", "合规", "检查",
        "safety", "moderate", "filter", "compliance", "guard",
    ],
    TaskType.EMBEDDING: [
        "嵌入", "向量", "相似度", "检索",
        "embed", "vector", "similarity", "search", "retrieval",
    ],
}

# 任务→逻辑模型路由表 (按优先级排序)
TASK_ROUTE_MAP: dict[TaskType, list[str]] = {
    TaskType.CODE: [
        "qwen3-coder", "deepseek-v4", "deepseek-v4-flash", "devstral",
        "codestral", "deepseek-coder", "glm-5", "qwen3.5",
        "claude-4-sonnet", "gpt-4o", "o3-mini",
    ],
    TaskType.REASONING: [
        "o3", "o4-mini", "deepseek-r1", "deepseek-v4", "claude-4-opus",
        "glm-5", "qwen3.5", "nemotron-ultra", "gemini-2.5-pro",
        "mistral-large", "gpt-4o",
    ],
    TaskType.VISION: [
        "gemini-2.5-pro", "gpt-4o", "o4-mini", "glm-5",
        "claude-4-sonnet", "llama3.2-vision",
    ],
    TaskType.CHAT: [
        "glm-5", "gpt-4o", "claude-4-sonnet", "deepseek-v4",
        "qwen3.5", "gemini-2.5-pro", "mistral-large",
        "deepseek-v4-flash", "glm-4", "mistral-small",
    ],
    TaskType.SAFETY: ["llama-guard"],
    TaskType.EMBEDDING: ["bge-m3"],
}


def classify_task(prompt: str) -> TaskType:
    """从 prompt 关键词推断任务类型"""
    prompt_lower = prompt.lower()
    scores: dict[TaskType, int] = defaultdict(int)
    for task_type, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                scores[task_type] += 1
    if not scores or max(scores.values()) == 0:
        return TaskType.CHAT
    return max(scores, key=scores.get)


# ─── 智能网关 ────────────────────────────────────────────────────────────

class SmartGateway:
    """LLM 智能网关 — 统一入口"""

    def __init__(
        self,
        balance_strategy: BalanceStrategy = BalanceStrategy.WEIGHTED,
        default_tier: ModelTier = ModelTier.STRONG,
        fingerprint_db: FingerprintDB | None = None,
    ):
        self.fingerprint_db = fingerprint_db or FingerprintDB()
        self.registry = ModelRegistry(self.fingerprint_db)
        self.balancer = LoadBalancer(self.registry, balance_strategy)
        self.default_tier = default_tier
        self._call_log: list[dict] = []

    @property
    def balance_strategy(self) -> BalanceStrategy:
        return self.balancer.default_strategy

    @balance_strategy.setter
    def balance_strategy(self, v: BalanceStrategy):
        self.balancer.default_strategy = v

    # ─── 模型管理 ───

    def add_channel(
        self,
        channel_id: str,
        channel_name: str,
        base_url: str,
        api_key: str,
        model_ids: list[str],
        priority: int = 0,
        weight: int = 1,
        free_models: set[str] | None = None,
    ) -> dict:
        """添加渠道及其模型列表"""
        return self.registry.register_channel(
            channel_id=channel_id,
            channel_name=channel_name,
            base_url=base_url,
            api_key=api_key,
            model_ids=model_ids,
            priority=priority,
            weight=weight,
            free_models=free_models,
        )

    def add_model(
        self,
        model_id: str,
        channel_id: str,
        channel_name: str = "",
        base_url: str = "",
        api_key: str = "",
        priority: int = 0,
        weight: int = 1,
        is_free: bool = False,
    ) -> str:
        """添加单个模型实例"""
        return self.registry.register(
            model_id=model_id,
            channel_id=channel_id,
            channel_name=channel_name,
            base_url=base_url,
            api_key=api_key,
            priority=priority,
            weight=weight,
            is_free=is_free,
        )

    # ─── 模型解析 ───

    def resolve_model(
        self,
        model: str,
        prompt: str = "",
        tier: ModelTier | None = None,
    ) -> dict:
        """
        解析模型请求 → 选择具体渠道实例

        三种模式:
          1. model="auto"     → 任务分类 → 最佳逻辑模型 → 均衡
          2. model="deepseek-v4" (逻辑名) → 跨渠道均衡
          3. model="deepseek-chat" (原始ID) → 指纹识别 → 归入逻辑模型 → 均衡

        Returns: {
            "canonical": str,          # 逻辑模型名
            "model_id": str,           # 实际发送给渠道的 model_id
            "channel_id": str,         # 渠道标识
            "channel_name": str,       # 渠道名
            "base_url": str,           # 渠道 API 地址
            "api_key": str,            # 渠道 API Key
            "task": str,               # 任务类型 (auto 时)
            "tier": str,               # 模型等级
            "strategy": str,           # 解析策略 (auto/explicit/fingerprint)
        }
        """
        tier = tier or self.default_tier

        # ── 模式 1: auto 智能分流 ──
        if model == "auto":
            return self._resolve_auto(prompt, tier)

        # ── 模式 2: 逻辑模型名 ──
        logical = self.registry.get(model)
        if logical:
            return self._resolve_logical(logical, tier, "explicit")

        # ── 模式 3: 原始 model_id → 指纹识别 ──
        canonical = self.fingerprint_db.canonical_name(model)
        logical = self.registry.get(canonical)
        if logical:
            return self._resolve_logical(logical, tier, "fingerprint")

        # ── 未找到 ──
        return {"error": f"model '{model}' not found in any channel"}

    def _resolve_auto(self, prompt: str, tier: ModelTier) -> dict:
        """auto 模式: 分类 → 路由 → 均衡"""
        task = classify_task(prompt)
        candidates = TASK_ROUTE_MAP.get(task, [])

        # 按优先级找第一个有实例的逻辑模型
        for canonical in candidates:
            logical = self.registry.get(canonical)
            if logical and logical.healthy_instances():
                inst = self.balancer.select(canonical)
                if inst:
                    return self._format_result(
                        inst, logical, task=task, strategy="auto"
                    )

        # 降级: 找同 tier 的任意模型
        for logical in self.registry.models.values():
            if logical.tier == tier and logical.healthy_instances():
                inst = self.balancer.select(logical.canonical)
                if inst:
                    return self._format_result(
                        inst, logical, task=task, strategy="auto-fallback"
                    )

        # 最终降级: 任意健康模型
        for logical in self.registry.models.values():
            if logical.healthy_instances():
                inst = self.balancer.select(logical.canonical)
                if inst:
                    return self._format_result(
                        inst, logical, task=task, strategy="auto-last-resort"
                    )

        return {"error": "no healthy model available"}

    def _resolve_logical(self, logical: LogicalModel, tier: ModelTier, strategy: str) -> dict:
        """逻辑模型名: 跨渠道均衡"""
        inst = self.balancer.select(logical.canonical)
        if not inst:
            return {"error": f"no healthy instance for '{logical.canonical}'"}
        return self._format_result(inst, logical, strategy=strategy)

    def _format_result(
        self,
        inst: ChannelInstance,
        logical: LogicalModel,
        task: TaskType = TaskType.CHAT,
        strategy: str = "explicit",
    ) -> dict:
        return {
            "canonical": logical.canonical,
            "model_id": inst.model_id,
            "channel_id": inst.channel_id,
            "channel_name": inst.channel_name,
            "base_url": inst.base_url,
            "api_key": inst.api_key,
            "task": task.value,
            "tier": logical.tier.value,
            "strategy": strategy,
            "is_free": inst.is_free,
            "instance_latency_ms": round(inst.latency_ms, 1),
            "instance_success_rate": round(inst.success_rate, 3),
        }

    # ─── 聊天调用 ───

    def chat(
        self,
        messages: list[dict],
        model: str = "auto",
        tier: ModelTier | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> dict:
        """
        同步聊天调用: 解析 → 转发 → 记指标 → 返回

        支持故障转移: 如果首选实例失败，自动切换到同逻辑模型的其他渠道。
        """
        prompt = messages[-1].get("content", "") if messages else ""
        failed_instances: set[str] = set()

        for attempt in range(max_retries):
            # 解析模型
            resolved = self.resolve_model(model, prompt=prompt, tier=tier)
            if "error" in resolved:
                return resolved

            inst_key = f"{resolved['channel_id']}:{resolved['model_id']}"
            if inst_key in failed_instances:
                # 跳过已失败的实例
                logical = self.registry.get(resolved["canonical"])
                if logical:
                    inst = self.balancer.select_for_call(
                        logical.canonical, exclude=failed_instances
                    )
                    if inst:
                        resolved = self._format_result(
                            inst, logical, task=TaskType(resolved.get("task", "chat")),
                            strategy=resolved["strategy"],
                        )
                    else:
                        return {"error": "all instances failed", "attempts": attempt + 1}
                else:
                    return resolved

            # 发起请求
            result, latency = self._do_request(resolved, messages, **kwargs)

            # 记录指标
            canonical = resolved["canonical"]
            logical = self.registry.get(canonical)
            if logical:
                for inst in logical.instances:
                    if inst.model_id == resolved["model_id"] and inst.channel_id == resolved["channel_id"]:
                        inst.record_call(latency, success=("error" not in result))
                        break

            # 成功则返回
            if "error" not in result:
                result["_gateway"] = {
                    "canonical": resolved["canonical"],
                    "model_id": resolved["model_id"],
                    "channel": resolved["channel_name"] or resolved["channel_id"],
                    "strategy": resolved["strategy"],
                    "task": resolved.get("task", ""),
                    "tier": resolved.get("tier", ""),
                    "latency_ms": round(latency, 1),
                    "attempt": attempt + 1,
                }
                return result

            # 失败则记录并重试
            failed_instances.add(inst_key)
            logger.warning(
                f"Attempt {attempt+1} failed: channel={resolved['channel_id']} "
                f"model={resolved['model_id']} error={result.get('error','')[:100]}"
            )

        return {"error": "max retries exceeded", "attempts": max_retries}

    def _do_request(
        self,
        resolved: dict,
        messages: list[dict],
        **kwargs,
    ) -> tuple[dict, float]:
        """执行实际 HTTP 请求"""
        url = f"{resolved['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {resolved['api_key']}",
            "Content-Type": "application/json",
        }
        body = {
            "model": resolved["model_id"],
            "messages": messages,
            **kwargs,
        }

        start = time.time()
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(url, json=body, headers=headers)
                    result = resp.json()
            else:
                data = json.dumps(body).encode()
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode())

            latency = (time.time() - start) * 1000
            return result, latency

        except Exception as e:
            latency = (time.time() - start) * 1000
            return {"error": str(e)}, latency

    # ─── 状态查询 ───

    def list_models(self) -> list[dict]:
        """列出所有可用模型 (OpenAI /v1/models 格式)"""
        models = [
            {
                "id": "auto",
                "object": "model",
                "created": 0,
                "owned_by": "smart-gateway",
            }
        ]
        for canonical in self.registry.all_canonicals():
            logical = self.registry.get(canonical)
            models.append({
                "id": canonical,
                "object": "model",
                "created": 0,
                "owned_by": logical.provider if logical else "unknown",
            })
        # 也保留原始 model_id 的映射
        for logical in self.registry.models.values():
            for inst in logical.instances:
                models.append({
                    "id": inst.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": inst.channel_id,
                })
        return models

    def stats(self) -> dict:
        """网关统计"""
        return self.registry.to_dict()

    def test_route(self, prompts: list[str]) -> list[dict]:
        """批量测试路由"""
        results = []
        for p in prompts:
            r = self.resolve_model("auto", prompt=p)
            results.append({
                "prompt": p[:50] + ("..." if len(p) > 50 else ""),
                **{k: v for k, v in r.items() if k != "api_key"},
            })
        return results
