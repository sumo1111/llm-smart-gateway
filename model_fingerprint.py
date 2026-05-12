"""
Phase 1: 模型指纹识别 — Model Fingerprint

核心问题：同一个底层模型在不同渠道(channel)有不同的 model_id 命名。
例如 DeepSeek V4 可能在不同渠道叫：
  - deepseek-chat          (渠道A: DeepSeek官方)
  - deepseek-v4-pro        (渠道B: 某代理商)
  - deepseek/deepseek-chat (渠道C: 另一个代理商)

本模块通过"指纹映射表"将这些别名统一归类到同一个逻辑模型。

设计原则：
  1. 指纹表可配置（YAML/JSON），支持用户自定义
  2. 内置常见模型的默认映射
  3. 支持模糊匹配（正则）+ 精确匹配
  4. 输出：原始 model_id → 逻辑模型名（canonical name）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelTier(str, Enum):
    """模型能力等级"""
    FLAGSHIP = "flagship"   # 最强推理
    STRONG = "strong"       # 强通用
    FAST = "fast"           # 快速
    FREE = "free"           # 免费层


class TaskType(str, Enum):
    """擅长任务类型"""
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    CHAT = "chat"
    EMBEDDING = "embedding"
    SAFETY = "safety"


@dataclass
class ModelFingerprint:
    """模型指纹 — 描述一个逻辑模型的身份"""
    canonical: str                          # 逻辑名 e.g. "deepseek-v4"
    provider: str                           # 原始厂商 e.g. "deepseek"
    tier: ModelTier = ModelTier.STRONG      # 能力等级
    tasks: set[TaskType] = field(default_factory=lambda: {TaskType.CHAT})
    context_length: int = 32768             # 上下文长度
    aliases: list[str] = field(default_factory=list)  # 精确别名列表
    patterns: list[str] = field(default_factory=list)  # 正则别名列表
    description: str = ""

    def match(self, model_id: str) -> bool:
        """检查一个原始 model_id 是否匹配此指纹"""
        mid = model_id.lower().strip()

        # 精确匹配
        for alias in self.aliases:
            if mid == alias.lower():
                return True

        # 正则匹配
        for pat in self.patterns:
            if re.search(pat, mid):
                return True

        return False


# ─── 内置指纹库 ──────────────────────────────────────────────────────────
# 覆盖主流大模型，每个 fingerprint 包含所有已知的渠道别名

BUILTIN_FINGERPRINTS: list[ModelFingerprint] = [
    # ═══ DeepSeek ═══
    ModelFingerprint(
        canonical="deepseek-v4",
        provider="deepseek",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.CODE, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "deepseek-chat", "deepseek-v4-pro", "deepseek-v4-chat",
            "deepseek/deepseek-chat", "deepseek-ai/deepseek-chat",
        ],
        patterns=[
            r"deepseek.?v?4.?pro", r"deepseek.?v?4.?chat",
            r"deepseek-ai/deepseek-v4-pro",
        ],
        description="DeepSeek V4 旗舰推理模型",
    ),
    ModelFingerprint(
        canonical="deepseek-v4-flash",
        provider="deepseek",
        tier=ModelTier.FAST,
        tasks={TaskType.CODE, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "deepseek-v4-flash", "deepseek-flash",
            "deepseek/deepseek-v4-flash",
        ],
        patterns=[
            r"deepseek.?v?4.?flash", r"deepseek-ai/deepseek-v4-flash",
        ],
        description="DeepSeek V4 Flash 快速版",
    ),
    ModelFingerprint(
        canonical="deepseek-r1",
        provider="deepseek",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING},
        context_length=65536,
        aliases=[
            "deepseek-reasoner", "deepseek-r1",
            "deepseek/deepseek-reasoner",
        ],
        patterns=[
            r"deepseek.?r1", r"deepseek.?reasoner",
        ],
        description="DeepSeek R1 推理专精模型",
    ),
    ModelFingerprint(
        canonical="deepseek-coder",
        provider="deepseek",
        tier=ModelTier.FAST,
        tasks={TaskType.CODE},
        context_length=65536,
        aliases=["deepseek-coder", "deepseek/deepseek-coder"],
        patterns=[r"deepseek.?coder"],
        description="DeepSeek Coder 代码模型",
    ),

    # ═══ GLM (智谱) ═══
    ModelFingerprint(
        canonical="glm-5",
        provider="zhipu",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.VISION, TaskType.CODE, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "glm-5", "glm5", "glm-5.1", "glm-5-plus",
            "z-ai/glm-5.1", "z-ai/glm5",
            "chatglm-5", "chatglm5",
        ],
        patterns=[
            r"glm.?5\.?1?", r"chatglm.?5",
        ],
        description="智谱 GLM-5 旗舰模型",
    ),
    ModelFingerprint(
        canonical="glm-4",
        provider="zhipu",
        tier=ModelTier.FAST,
        tasks={TaskType.CHAT},
        context_length=65536,
        aliases=[
            "glm-4", "glm4", "glm-4-air", "glm-4.5-air",
            "glm-4-flash", "glm-4-plus",
            "z-ai/glm-4.5-air",
        ],
        patterns=[
            r"glm.?4(\.5)?.?air", r"glm.?4.?flash", r"glm.?4\.?7",
        ],
        description="智谱 GLM-4 快速模型",
    ),

    # ═══ Qwen (通义) ═══
    ModelFingerprint(
        canonical="qwen3.5",
        provider="alibaba",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.CODE, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "qwen3.5", "qwen-3.5", "qwen3.5-397b",
            "qwen/qwen3.5-397b-a17b",
        ],
        patterns=[
            r"qwen.?3\.5", r"qwen/qwen3\.5",
        ],
        description="通义千问 3.5 旗舰模型",
    ),
    ModelFingerprint(
        canonical="qwen3-coder",
        provider="alibaba",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.CODE},
        context_length=131072,
        aliases=[
            "qwen3-coder", "qwen-coder",
            "qwen/qwen3-coder-480b",
        ],
        patterns=[
            r"qwen.?3.?coder.?480", r"qwen3-coder-480b",
        ],
        description="通义千问 3 Coder 代码旗舰",
    ),
    ModelFingerprint(
        canonical="qwen3-coder-free",
        provider="alibaba",
        tier=ModelTier.FREE,
        tasks={TaskType.CODE},
        context_length=32768,
        aliases=["qwen3-coder:free", "qwen/qwen3-coder:free"],
        patterns=[r"qwen.?3.?coder.*free"],
        description="通义千问 3 Coder 免费版",
    ),
    ModelFingerprint(
        canonical="qwen2.5-coder",
        provider="alibaba",
        tier=ModelTier.FAST,
        tasks={TaskType.CODE},
        context_length=65536,
        aliases=["qwen2.5-coder-32b", "qwen/qwen2.5-coder-32b-instruct"],
        patterns=[r"qwen.?2\.5.?coder"],
        description="通义千问 2.5 Coder",
    ),

    # ═══ Llama ═══
    ModelFingerprint(
        canonical="llama4-maverick",
        provider="meta",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=131072,
        aliases=[
            "llama-4-maverick", "llama4-maverick",
            "meta/llama-4-maverick-17b-128e-instruct",
        ],
        patterns=[r"llama.?4.?maverick"],
        description="Meta Llama 4 Maverick",
    ),
    ModelFingerprint(
        canonical="llama3.3-70b",
        provider="meta",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=65536,
        aliases=[
            "llama-3.3-70b", "llama3.3-70b-instruct",
            "meta/llama-3.3-70b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "dracarys-70b",
        ],
        patterns=[
            r"llama.?3\.3.?70b", r"dracarys",
        ],
        description="Meta Llama 3.3 70B",
    ),
    ModelFingerprint(
        canonical="llama3.1-70b",
        provider="meta",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=65536,
        aliases=[
            "llama-3.1-70b", "llama3.1-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
        ],
        patterns=[r"llama.?3\.1.?70b"],
        description="Meta Llama 3.1 70B",
    ),
    ModelFingerprint(
        canonical="llama3.1-8b",
        provider="meta",
        tier=ModelTier.FAST,
        tasks={TaskType.CHAT},
        context_length=32768,
        aliases=[
            "llama-3.1-8b", "llama3.1-8b-instruct",
            "meta/llama-3.1-8b-instruct",
        ],
        patterns=[r"llama.?3\.1.?8b"],
        description="Meta Llama 3.1 8B",
    ),
    ModelFingerprint(
        canonical="llama3-70b",
        provider="meta",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT},
        context_length=32768,
        aliases=["llama-3-70b", "llama3-70b-instruct"],
        patterns=[r"llama.?3[^.\d].?70b"],
        description="Meta Llama 3 70B",
    ),
    ModelFingerprint(
        canonical="llama3.2-vision",
        provider="meta",
        tier=ModelTier.STRONG,
        tasks={TaskType.VISION, TaskType.CHAT},
        context_length=65536,
        aliases=[
            "llama-3.2-90b-vision", "llama-3.2-11b-vision",
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
        ],
        patterns=[r"llama.?3\.2.*vision"],
        description="Meta Llama 3.2 Vision",
    ),

    # ═══ Nemotron (NVIDIA) ═══
    ModelFingerprint(
        canonical="nemotron-ultra",
        provider="nvidia",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING},
        context_length=65536,
        aliases=[
            "nemotron-ultra", "llama-3.1-nemotron-ultra-253b",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        ],
        patterns=[r"nemotron.?ultra"],
        description="NVIDIA Nemotron Ultra 253B 旗舰推理",
    ),
    ModelFingerprint(
        canonical="nemotron-super",
        provider="nvidia",
        tier=ModelTier.STRONG,
        tasks={TaskType.REASONING, TaskType.CHAT},
        context_length=65536,
        aliases=[
            "nemotron-super", "llama-3.3-nemotron-super-49b",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "nemotron-3-super",
        ],
        patterns=[r"nemotron.?3?.?super"],
        description="NVIDIA Nemotron Super 49B",
    ),

    # ═══ Mistral ═══
    ModelFingerprint(
        canonical="mistral-large",
        provider="mistral",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "mistral-large", "mistral-large-latest",
            "mistral-large-3", "mistral-large-3-675b",
            "open-mistral-large",
        ],
        patterns=[
            r"mistral.?large",
        ],
        description="Mistral Large 旗舰模型",
    ),
    ModelFingerprint(
        canonical="mistral-medium",
        provider="mistral",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT},
        context_length=65536,
        aliases=[
            "mistral-medium", "mistral-medium-3",
            "mistralai/mistral-medium-3-instruct",
        ],
        patterns=[r"mistral.?medium"],
        description="Mistral Medium",
    ),
    ModelFingerprint(
        canonical="mistral-small",
        provider="mistral",
        tier=ModelTier.FAST,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=65536,
        aliases=[
            "mistral-small", "mistral-small-latest",
            "mistralai/mistral-small-4-119b-2603",
        ],
        patterns=[r"mistral.?small"],
        description="Mistral Small 快速模型",
    ),
    ModelFingerprint(
        canonical="codestral",
        provider="mistral",
        tier=ModelTier.STRONG,
        tasks={TaskType.CODE},
        context_length=65536,
        aliases=[
            "codestral", "codestral-latest",
            "mistralai/codestral-22b-instruct-v0.1",
        ],
        patterns=[r"codestral"],
        description="Mistral Codestral 代码模型",
    ),
    ModelFingerprint(
        canonical="devstral",
        provider="mistral",
        tier=ModelTier.STRONG,
        tasks={TaskType.CODE},
        context_length=65536,
        aliases=["devstral", "mistralai/devstral-2-123b-instruct-2512"],
        patterns=[r"devstral"],
        description="Mistral Devstral 开发者模型",
    ),

    # ═══ Gemini ═══
    ModelFingerprint(
        canonical="gemini-2.5-pro",
        provider="google",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.VISION, TaskType.CODE, TaskType.CHAT},
        context_length=1048576,
        aliases=[
            "gemini-2.5-pro", "gemini-2.5-pro-preview",
            "google/gemini-2.5-pro-preview-06-05",
        ],
        patterns=[r"gemini.?2\.5.?pro"],
        description="Google Gemini 2.5 Pro 旗舰",
    ),
    ModelFingerprint(
        canonical="gemini-2.5-flash",
        provider="google",
        tier=ModelTier.FAST,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=1048576,
        aliases=[
            "gemini-2.5-flash", "gemini-2.5-flash-preview",
        ],
        patterns=[r"gemini.?2\.5.?flash"],
        description="Google Gemini 2.5 Flash 快速版",
    ),

    # ═══ GPT ═══
    ModelFingerprint(
        canonical="gpt-4o",
        provider="openai",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.VISION, TaskType.CODE, TaskType.CHAT},
        context_length=131072,
        aliases=[
            "gpt-4o", "gpt-4o-latest", "gpt-4o-2024-11-20",
        ],
        patterns=[r"gpt.?4o(?!mini)"],
        description="OpenAI GPT-4o 旗舰",
    ),
    ModelFingerprint(
        canonical="gpt-4o-mini",
        provider="openai",
        tier=ModelTier.FAST,
        tasks={TaskType.CHAT, TaskType.CODE},
        context_length=131072,
        aliases=["gpt-4o-mini", "gpt-4o-mini-latest"],
        patterns=[r"gpt.?4o.?mini"],
        description="OpenAI GPT-4o Mini 快速版",
    ),
    ModelFingerprint(
        canonical="o3",
        provider="openai",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING},
        context_length=200000,
        aliases=["o3", "o3-latest"],
        patterns=[r"\bo3\b(?!mini)"],
        description="OpenAI o3 推理模型",
    ),
    ModelFingerprint(
        canonical="o3-mini",
        provider="openai",
        tier=ModelTier.STRONG,
        tasks={TaskType.REASONING, TaskType.CODE},
        context_length=200000,
        aliases=["o3-mini", "o3-mini-latest"],
        patterns=[r"o3.?mini"],
        description="OpenAI o3-mini 推理模型",
    ),
    ModelFingerprint(
        canonical="o4-mini",
        provider="openai",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.VISION, TaskType.CODE},
        context_length=200000,
        aliases=["o4-mini", "o4-mini-latest"],
        patterns=[r"o4.?mini"],
        description="OpenAI o4-mini 多模态推理",
    ),

    # ═══ Claude ═══
    ModelFingerprint(
        canonical="claude-4-sonnet",
        provider="anthropic",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.CODE, TaskType.CHAT},
        context_length=200000,
        aliases=[
            "claude-4-sonnet", "claude-sonnet-4",
            "claude-4-sonnet-20250514",
        ],
        patterns=[r"claude.?4.?sonnet|claude.?sonnet.?4"],
        description="Anthropic Claude 4 Sonnet",
    ),
    ModelFingerprint(
        canonical="claude-4-opus",
        provider="anthropic",
        tier=ModelTier.FLAGSHIP,
        tasks={TaskType.REASONING, TaskType.CHAT},
        context_length=200000,
        aliases=["claude-4-opus", "claude-opus-4"],
        patterns=[r"claude.?4.?opus|claude.?opus.?4"],
        description="Anthropic Claude 4 Opus",
    ),

    # ═══ 其他常见模型 ═══
    ModelFingerprint(
        canonical="yi-large",
        provider="01ai",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT},
        aliases=["yi-large", "01-ai/yi-large", "yi-large-rag"],
        patterns=[r"yi.?large"],
        description="零一万物 Yi Large",
    ),
    ModelFingerprint(
        canonical="kimi-k2",
        provider="moonshot",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT, TaskType.CODE},
        aliases=["kimi-k2", "moonshotai/kimi-k2-instruct"],
        patterns=[r"kimi.?k2"],
        description="月之暗面 Kimi K2",
    ),
    ModelFingerprint(
        canonical="mixtral-8x22b",
        provider="mistral",
        tier=ModelTier.STRONG,
        tasks={TaskType.CHAT, TaskType.CODE},
        aliases=["mixtral-8x22b", "mistralai/mixtral-8x22b-instruct"],
        patterns=[r"mixtral.?8x22b"],
        description="Mixtral 8x22B MoE",
    ),
    ModelFingerprint(
        canonical="hermes-3",
        provider="nous",
        tier=ModelTier.FREE,
        tasks={TaskType.CHAT, TaskType.CODE},
        aliases=["hermes-3-llama-3.1-405b", "nousresearch/hermes-3-llama-3.1-405b"],
        patterns=[r"hermes.?3"],
        description="Nous Hermes 3 (Llama 405B 基座)",
    ),
    ModelFingerprint(
        canonical="dolphin-mistral",
        provider="cognitivecomputations",
        tier=ModelTier.FREE,
        tasks={TaskType.CHAT},
        aliases=["dolphin-mistral", "cognitivecomputations/dolphin-mistral-24b"],
        patterns=[r"dolphin.?mistral"],
        description="Dolphin Mistral (去审查版)",
    ),
]


class FingerprintDB:
    """指纹数据库 — 管理 + 查询"""

    def __init__(self):
        self.fingerprints: list[ModelFingerprint] = list(BUILTIN_FINGERPRINTS)
        self._cache: dict[str, str] = {}  # model_id → canonical (缓存)

    def add(self, fp: ModelFingerprint):
        """添加自定义指纹"""
        self.fingerprints.append(fp)
        self._cache.clear()  # 清缓存

    def remove(self, canonical: str):
        """移除指纹"""
        self.fingerprints = [f for f in self.fingerprints if f.canonical != canonical]
        self._cache.clear()

    def identify(self, model_id: str) -> Optional[ModelFingerprint]:
        """识别一个原始 model_id 对应的指纹"""
        if model_id in self._cache:
            canonical = self._cache[model_id]
            return next((f for f in self.fingerprints if f.canonical == canonical), None)

        for fp in self.fingerprints:
            if fp.match(model_id):
                self._cache[model_id] = fp.canonical
                return fp

        return None

    def canonical_name(self, model_id: str) -> str:
        """获取逻辑名，未匹配则返回原名"""
        fp = self.identify(model_id)
        return fp.canonical if fp else model_id

    def load_from_file(self, path: str):
        """从 JSON 文件加载自定义指纹"""
        with open(path) as f:
            data = json.load(f)
        for item in data:
            fp = ModelFingerprint(
                canonical=item["canonical"],
                provider=item.get("provider", "unknown"),
                tier=ModelTier(item.get("tier", "strong")),
                tasks={TaskType(t) for t in item.get("tasks", ["chat"])},
                context_length=item.get("context_length", 32768),
                aliases=item.get("aliases", []),
                patterns=item.get("patterns", []),
                description=item.get("description", ""),
            )
            self.add(fp)

    def export_builtin(self, path: str):
        """导出内置指纹为 JSON (方便用户修改)"""
        data = []
        for fp in self.fingerprints:
            data.append({
                "canonical": fp.canonical,
                "provider": fp.provider,
                "tier": fp.tier.value,
                "tasks": [t.value for t in fp.tasks],
                "context_length": fp.context_length,
                "aliases": fp.aliases,
                "patterns": fp.patterns,
                "description": fp.description,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def summary(self) -> dict:
        return {
            "total_fingerprints": len(self.fingerprints),
            "total_aliases": sum(len(f.aliases) for f in self.fingerprints),
            "total_patterns": sum(len(f.patterns) for f in self.fingerprints),
            "providers": sorted(set(f.provider for f in self.fingerprints)),
            "canonicals": [f.canonical for f in self.fingerprints],
        }


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="模型指纹识别工具")
    sub = parser.add_subparsers(dest="cmd")

    id_p = sub.add_parser("identify", help="识别模型 ID")
    id_p.add_argument("model_ids", nargs="+", help="原始模型 ID")

    sub.add_parser("list", help="列出所有内置指纹")

    export_p = sub.add_parser("export", help="导出内置指纹为 JSON")
    export_p.add_argument("-o", "--output", default="fingerprints.json")

    args = parser.parse_args()
    db = FingerprintDB()

    if args.cmd == "identify":
        for mid in args.model_ids:
            fp = db.identify(mid)
            if fp:
                print(f"✅ {mid}")
                print(f"   → 逻辑名: {fp.canonical}")
                print(f"   → 厂商: {fp.provider}")
                print(f"   → 等级: {fp.tier.value}")
                print(f"   → 任务: {[t.value for t in fp.tasks]}")
                print(f"   → 上下文: {fp.context_length}")
            else:
                print(f"❓ {mid} → 未匹配 (将保留原名)")

    elif args.cmd == "list":
        for fp in db.fingerprints:
            print(f"{fp.canonical:<25} [{fp.provider:<12}] tier={fp.tier.value:<10} aliases={len(fp.aliases)} patterns={len(fp.patterns)}")

    elif args.cmd == "export":
        db.export_builtin(args.output)
        print(f"✅ 已导出到 {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
