"""
Phase 5: one-api 适配器 — 从 one-api / new-api 实例同步渠道和模型

one-api 的数据结构:
  - Channel: 渠道 (含 base_url, api_key, models 列表, 优先级等)
  - 模型列表: 每个渠道支持的 model_id 列表 (逗号分隔)

本模块从 one-api 的数据库/API 拉取渠道信息，注册到 SmartGateway。
"""

from __future__ import annotations

import json
import os
from typing import Optional

from gateway import SmartGateway
from model_fingerprint import FingerprintDB


class OneAPIAdapter:
    """one-api 适配器 — 桥接 one-api 和 SmartGateway"""

    def __init__(self, gateway: SmartGateway):
        self.gateway = gateway

    @staticmethod
    def parse_model_string(model_str: str) -> list[str]:
        """解析 one-api 的模型字符串 (逗号/换行分隔)"""
        models = []
        for part in model_str.replace("\n", ",").split(","):
            m = part.strip()
            if m and not m.startswith("#"):
                models.append(m)
        return models

    def load_from_json(self, path: str) -> dict:
        """
        从 one-api 导出的 JSON 文件加载渠道配置。

        预期格式 (one-api 数据库 channels 表导出):
        [
          {
            "id": 1,
            "name": "DeepSeek 官方",
            "base_url": "https://api.deepseek.com",
            "key": "sk-xxx",
            "models": "deepseek-chat,deepseek-reasoner,deepseek-coder",
            "priority": 10,
            "weight": 1,
            "status": 1
          },
          ...
        ]
        """
        with open(path) as f:
            channels = json.load(f)
        return self._load_channels(channels)

    def load_from_api(
        self,
        base_url: str,
        admin_token: str,
    ) -> dict:
        """
        从 one-api 管理接口拉取渠道列表。

        one-api API: GET /api/channel/?p=0&pagesize=100
        需要 admin token。
        """
        try:
            import httpx
            url = f"{base_url.rstrip('/')}/api/channel/"
            headers = {
                "Authorization": f"Bearer {admin_token}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, headers=headers, params={"p": 0, "pagesize": 100})
                data = resp.json()
        except ImportError:
            import urllib.request
            url = f"{base_url.rstrip('/')}/api/channel/?p=0&pagesize=100"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

        channels = data.get("data", []) if isinstance(data, dict) else data
        return self._load_channels(channels)

    def _load_channels(self, channels: list[dict]) -> dict:
        """批量加载渠道到网关"""
        result = {
            "channels_loaded": 0,
            "total_models_registered": 0,
            "total_models_merged": 0,
            "logical_models": 0,
            "details": [],
        }

        for ch in channels:
            if ch.get("status", 1) != 1:
                continue  # 跳过禁用的渠道

            channel_id = str(ch.get("id", ch.get("channel_id", "")))
            channel_name = ch.get("name", ch.get("channel_name", channel_id))
            base_url = ch.get("base_url", "")
            api_key = ch.get("key", ch.get("api_key", ch.get("apikey", "")))
            models_str = ch.get("models", "")
            priority = ch.get("priority", 0)
            weight = ch.get("weight", 1)

            model_ids = self.parse_model_string(models_str)
            if not model_ids:
                continue

            reg = self.gateway.add_channel(
                channel_id=f"oneapi-{channel_id}",
                channel_name=channel_name,
                base_url=base_url,
                api_key=api_key,
                model_ids=model_ids,
                priority=priority,
                weight=weight,
            )

            result["channels_loaded"] += 1
            result["total_models_registered"] += reg["registered"]
            result["total_models_merged"] += reg["merged"]
            result["details"].append({
                "channel": channel_name,
                "models_in": len(model_ids),
                "merged": reg["merged"],
                "unique_canonicals": reg["unique_canonicals"],
            })

        result["logical_models"] = len(self.gateway.registry.models)
        return result

    def load_from_env(self) -> dict:
        """
        从环境变量加载 one-api 连接信息。

        环境变量:
          ONE_API_BASE_URL — one-api 地址 e.g. http://localhost:3000
          ONE_API_ADMIN_TOKEN — one-api admin token
          或
          ONE_API_CHANNELS_JSON — 渠道配置 JSON 文件路径
        """
        json_path = os.getenv("ONE_API_CHANNELS_JSON", "")
        if json_path and os.path.exists(json_path):
            return self.load_from_json(json_path)

        base_url = os.getenv("ONE_API_BASE_URL", "")
        token = os.getenv("ONE_API_ADMIN_TOKEN", "")
        if base_url and token:
            return self.load_from_api(base_url, token)

        return {"error": "no one-api config found in env"}
