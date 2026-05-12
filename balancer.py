"""
Phase 3: 跨渠道负载均衡器 — Cross-Channel Load Balancer

在逻辑模型注册表基础上，对同一逻辑模型下的多渠道实例进行智能均衡。
支持：加权轮询、最闲优先、最快优先、优先级优先、故障转移+熔断恢复。
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Optional

from registry import ChannelInstance, LogicalModel, ModelRegistry
from model_fingerprint import TaskType, ModelTier


class BalanceStrategy(str, Enum):
    WEIGHTED = "weighted"               # 加权随机 (综合权重)
    ROUND_ROBIN = "round_robin"         # 轮询
    LEAST_LOADED = "least_loaded"       # 最久未用优先
    FASTEST = "fastest"                 # 最低延迟优先
    PRIORITY = "priority"               # 按渠道优先级
    FAILOVER = "failover"               # 优先级 + 故障转移


class LoadBalancer:
    """跨渠道负载均衡器"""

    def __init__(
        self,
        registry: ModelRegistry,
        default_strategy: BalanceStrategy = BalanceStrategy.WEIGHTED,
        failover_max_retries: int = 3,
    ):
        self.registry = registry
        self.default_strategy = default_strategy
        self.failover_max_retries = failover_max_retries
        self._rr_counters: dict[str, int] = {}  # round-robin 计数器

    def select(
        self,
        canonical: str,
        strategy: BalanceStrategy | None = None,
        prefer_free: bool = False,
    ) -> Optional[ChannelInstance]:
        """
        从逻辑模型的渠道实例中选择一个。

        Args:
            canonical: 逻辑模型名 e.g. "deepseek-v4"
            strategy: 均衡策略 (默认用 default_strategy)
            prefer_free: 偏好免费实例

        Returns:
            选中的渠道实例，或 None (无可用实例)
        """
        model = self.registry.get(canonical)
        if not model:
            return None

        candidates = model.healthy_instances()
        if not candidates:
            # 所有实例都不健康，尝试熔断恢复 (降级返回最不差的)
            candidates = model.instances
            if not candidates:
                return None

        if prefer_free:
            free = [i for i in candidates if i.is_free]
            if free:
                candidates = free

        strategy = strategy or self.default_strategy

        if strategy == BalanceStrategy.WEIGHTED:
            return self._weighted_select(candidates)
        elif strategy == BalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(canonical, candidates)
        elif strategy == BalanceStrategy.LEAST_LOADED:
            return self._least_loaded_select(candidates)
        elif strategy == BalanceStrategy.FASTEST:
            return self._fastest_select(candidates)
        elif strategy == BalanceStrategy.PRIORITY:
            return self._priority_select(candidates)
        elif strategy == BalanceStrategy.FAILOVER:
            return self._failover_select(canonical, candidates)
        else:
            return self._weighted_select(candidates)

    def select_for_call(
        self,
        canonical: str,
        strategy: BalanceStrategy | None = None,
        prefer_free: bool = False,
        exclude: set[str] | None = None,
    ) -> Optional[ChannelInstance]:
        """
        选择实例用于实际调用 (支持排除已失败的实例)。
        如果首选实例失败，可以换 exclude 集合再次调用。
        """
        model = self.registry.get(canonical)
        if not model:
            return None

        candidates = model.healthy_instances()
        if not candidates:
            candidates = model.instances
        if not candidates:
            return None

        # 排除已失败实例
        if exclude:
            candidates = [i for i in candidates
                          if f"{i.channel_id}:{i.model_id}" not in exclude]
        if not candidates:
            return None

        if prefer_free:
            free = [i for i in candidates if i.is_free]
            if free:
                candidates = free

        strategy = strategy or self.default_strategy

        if strategy == BalanceStrategy.FAILOVER:
            return self._failover_select(canonical, candidates)
        else:
            return self.select(canonical, strategy, prefer_free)

    # ─── 具体策略实现 ───

    def _weighted_select(self, candidates: list[ChannelInstance]) -> ChannelInstance:
        """加权随机 — 综合权重(静态×成功率×延迟倒数)"""
        weights = [i.effective_weight() for i in candidates]
        total = sum(weights)
        if total <= 0:
            return random.choice(candidates)
        r = random.uniform(0, total)
        cum = 0
        for inst, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                return inst
        return candidates[-1]

    def _round_robin_select(self, canonical: str, candidates: list[ChannelInstance]) -> ChannelInstance:
        """轮询"""
        idx = self._rr_counters.get(canonical, 0)
        selected = candidates[idx % len(candidates)]
        self._rr_counters[canonical] = idx + 1
        return selected

    def _least_loaded_select(self, candidates: list[ChannelInstance]) -> ChannelInstance:
        """最久未用优先"""
        return min(candidates, key=lambda i: i.last_used)

    def _fastest_select(self, candidates: list[ChannelInstance]) -> ChannelInstance:
        """最低延迟优先"""
        return min(candidates, key=lambda i: i.latency_ms if i.total_calls > 0 else 99999)

    def _priority_select(self, candidates: list[ChannelInstance]) -> ChannelInstance:
        """按渠道优先级"""
        sorted_c = sorted(candidates, key=lambda i: (-i.priority, -i.effective_weight()))
        return sorted_c[0]

    def _failover_select(self, canonical: str, candidates: list[ChannelInstance]) -> ChannelInstance:
        """优先级 + 故障转移：先按优先级排，再按健康+权重"""
        sorted_c = sorted(candidates, key=lambda i: (
            not i.is_healthy,       # 健康优先
            -i.priority,            # 高优先级优先
            -i.effective_weight(),  # 高权重优先
        ))
        return sorted_c[0]
