"""
Phase 2: 逻辑模型注册表 — Logical Model Registry

在指纹识别基础上，将不同渠道的同名模型合并为逻辑端点，
每个逻辑端点下挂多个渠道实例，为跨渠道负载均衡提供基础。

核心概念:
  - ChannelInstance: 某个渠道上的一个物理模型实例
  - LogicalModel: 合并后的逻辑模型，包含多个渠道实例
  - ModelRegistry: 全局注册表，管理所有逻辑模型
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

from model_fingerprint import (
    FingerprintDB, ModelFingerprint, ModelTier, TaskType
)


@dataclass
class ChannelInstance:
    """渠道实例 — 一个物理模型在某个渠道上的端点"""
    model_id: str           # 该渠道上的原始 model_id e.g. "deepseek-chat"
    channel_id: str         # 渠道标识 e.g. "deepseek-official"
    channel_name: str = ""  # 渠道可读名 e.g. "DeepSeek 官方渠道"
    base_url: str = ""      # 渠道 API 地址 e.g. "https://api.deepseek.com/v1"
    api_key: str = ""       # 渠道 API Key
    priority: int = 0       # 渠道优先级 (越高越优先)
    weight: int = 1         # 负载均衡权重
    is_free: bool = False   # 是否免费

    # ─── 运行时指标 (线程安全) ───
    _total_calls: int = 0
    _success_calls: int = 0
    _fail_calls: int = 0
    _total_latency_ms: float = 0.0
    _last_latency_ms: float = 0.0
    _ema_latency_ms: float = 0.0
    _ema_success_rate: float = 1.0
    _last_used: float = 0.0
    _consecutive_fails: int = 0
    _disabled_until: float = 0.0  # 熔断: 禁用到此时间戳
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._total_calls

    @property
    def success_rate(self) -> float:
        with self._lock:
            return self._ema_success_rate

    @property
    def latency_ms(self) -> float:
        with self._lock:
            return self._ema_latency_ms

    @property
    def last_used(self) -> float:
        with self._lock:
            return self._last_used

    @property
    def is_healthy(self) -> bool:
        """是否健康 (未被熔断)"""
        with self._lock:
            return time.time() >= self._disabled_until

    def record_call(self, latency_ms: float, success: bool):
        """记录一次调用结果，更新指标"""
        with self._lock:
            self._total_calls += 1
            self._last_used = time.time()
            self._last_latency_ms = latency_ms

            if self._total_calls == 1:
                self._ema_latency_ms = latency_ms
            else:
                self._ema_latency_ms = self._ema_latency_ms * 0.7 + latency_ms * 0.3

            if success:
                self._success_calls += 1
                self._consecutive_fails = 0
                self._ema_success_rate = self._ema_success_rate * 0.95 + 1.0 * 0.05
            else:
                self._fail_calls += 1
                self._consecutive_fails += 1
                self._ema_success_rate = self._ema_success_rate * 0.95 + 0.0 * 0.05

                # 熔断: 连续失败 3 次禁用 30 秒
                if self._consecutive_fails >= 3:
                    self._disabled_until = time.time() + 30

    def effective_weight(self) -> float:
        """综合权重 = 静态权重 × 成功率 × 延迟倒数 × 健康状态"""
        if not self.is_healthy:
            return 0.0
        latency_factor = 1.0 / (1.0 + self._ema_latency_ms / 1000.0)
        return self.weight * self._ema_success_rate * latency_factor

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "base_url": self.base_url,
            "priority": self.priority,
            "weight": self.weight,
            "is_free": self.is_free,
            "healthy": self.is_healthy,
            "calls": self.total_calls,
            "success_rate": round(self.success_rate, 3),
            "latency_ms": round(self.latency_ms, 1),
            "last_used": self.last_used,
        }


@dataclass
class LogicalModel:
    """逻辑模型 — 合并后的统一端点"""
    canonical: str                          # 逻辑名 e.g. "deepseek-v4"
    fingerprint: Optional[ModelFingerprint] = None
    instances: list[ChannelInstance] = field(default_factory=list)

    @property
    def tier(self) -> ModelTier:
        return self.fingerprint.tier if self.fingerprint else ModelTier.STRONG

    @property
    def tasks(self) -> set[TaskType]:
        return self.fingerprint.tasks if self.fingerprint else {TaskType.CHAT}

    @property
    def provider(self) -> str:
        return self.fingerprint.provider if self.fingerprint else "unknown"

    @property
    def context_length(self) -> int:
        return self.fingerprint.context_length if self.fingerprint else 32768

    def healthy_instances(self) -> list[ChannelInstance]:
        """返回所有健康实例"""
        return [i for i in self.instances if i.is_healthy]

    def add_instance(self, inst: ChannelInstance):
        """添加渠道实例 (去重)"""
        for existing in self.instances:
            if existing.model_id == inst.model_id and existing.channel_id == inst.channel_id:
                # 更新已有实例的连接信息
                existing.base_url = inst.base_url or existing.base_url
                existing.api_key = inst.api_key or existing.api_key
                existing.priority = inst.priority if inst.priority else existing.priority
                existing.weight = inst.weight if inst.weight != 1 else existing.weight
                return
        self.instances.append(inst)

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "provider": self.provider,
            "tier": self.tier.value,
            "tasks": [t.value for t in self.tasks],
            "context_length": self.context_length,
            "instance_count": len(self.instances),
            "healthy_count": len(self.healthy_instances()),
            "instances": [i.to_dict() for i in self.instances],
        }


class ModelRegistry:
    """逻辑模型注册表"""

    def __init__(self, fingerprint_db: FingerprintDB | None = None):
        self.fingerprint_db = fingerprint_db or FingerprintDB()
        self.models: dict[str, LogicalModel] = {}  # canonical → LogicalModel
        self._unmatched: list[dict] = []  # 未匹配的模型记录

    def register(
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
        """
        注册一个渠道上的模型实例。
        自动通过指纹识别归类到逻辑模型。
        返回逻辑模型名 (canonical)。
        """
        fp = self.fingerprint_db.identify(model_id)
        canonical = fp.canonical if fp else model_id

        inst = ChannelInstance(
            model_id=model_id,
            channel_id=channel_id,
            channel_name=channel_name,
            base_url=base_url,
            api_key=api_key,
            priority=priority,
            weight=weight,
            is_free=is_free,
        )

        if canonical not in self.models:
            self.models[canonical] = LogicalModel(
                canonical=canonical,
                fingerprint=fp,
            )

        self.models[canonical].add_instance(inst)

        if not fp:
            self._unmatched.append({
                "model_id": model_id,
                "channel_id": channel_id,
            })

        return canonical

    def register_channel(
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
        """
        批量注册一个渠道的所有模型。

        Returns: {"channel_id": str, "registered": int, "merged": int}
        """
        free_models = free_models or set()
        registered = 0
        merged = 0

        for mid in model_ids:
            canonical = self.register(
                model_id=mid,
                channel_id=channel_id,
                channel_name=channel_name,
                base_url=base_url,
                api_key=api_key,
                priority=priority,
                weight=weight,
                is_free=(mid in free_models or mid.endswith(":free")),
            )
            registered += 1
            if canonical != mid:
                merged += 1

        return {
            "channel_id": channel_id,
            "registered": registered,
            "merged": merged,
            "unique_canonicals": len(set(
                self.fingerprint_db.canonical_name(mid) for mid in model_ids
            )),
        }

    def get(self, canonical: str) -> Optional[LogicalModel]:
        return self.models.get(canonical)

    def all_canonicals(self) -> list[str]:
        return sorted(self.models.keys())

    def models_by_tier(self, tier: ModelTier) -> list[LogicalModel]:
        return [m for m in self.models.values() if m.tier == tier]

    def models_by_task(self, task: TaskType) -> list[LogicalModel]:
        return [m for m in self.models.values() if task in m.tasks]

    def summary(self) -> dict:
        total_instances = sum(len(m.instances) for m in self.models.values())
        healthy_instances = sum(len(m.healthy_instances()) for m in self.models.values())
        by_tier = {}
        for tier in ModelTier:
            by_tier[tier.value] = len(self.models_by_tier(tier))
        by_task = {}
        for task in TaskType:
            by_task[task.value] = len(self.models_by_task(task))

        return {
            "logical_models": len(self.models),
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "unmatched_models": len(self._unmatched),
            "by_tier": by_tier,
            "by_task": by_task,
            "merge_ratio": f"{total_instances} → {len(self.models)} "
                           f"({(1 - len(self.models) / max(total_instances, 1)) * 100:.1f}% 去重)",
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "models": {k: v.to_dict() for k, v in sorted(self.models.items())},
            "unmatched": self._unmatched[:20],
        }
