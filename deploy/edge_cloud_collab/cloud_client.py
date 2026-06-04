"""
云端大模型客户端 (Cloud Client)

通过 openrouter API 调用 qwen3-vl-8B，实现协同纠错。
支持两种模式：
  - real: 实际调用 NIM API
  - mock: 模拟云端（用于本地开发和仿真）
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from openai import OpenAI

from .utils import detections_to_json, image_to_base64, parse_cloud_response


class CloudClient:
    """
    云端大模型客户端。

    Usage:
        client = CloudClient.from_config(config["cloud"])
        result = client.correct(image, local_dets, names)
    """

    def __init__(
        self,
        api_base: str,
        api_key: str | None = None,
        model: str = "meta/llama-3.2-11b-vision-instruct",
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        mock: bool = False,
        mock_latency_ms: tuple[float, float] = (200.0, 350.0),
        mock_boost: float = 0.12,
    ):
        """
        Args:
            api_base: API endpoint URL
            api_key: NVIDIA API key（为 None 时从环境变量 NIM_API_KEY 读取）
            model: 模型名称
            system_prompt: 系统提示词
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            timeout: API 超时时间
            mock: 是否使用模拟模式（不实际调用 API）
            mock_latency_ms: 模拟延迟范围 (min, max)
            mock_boost: 模拟模式下假设的 mAP 提升比例（仅用于仿真统计）
        """
        self.api_base = api_base
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.mock = mock
        self.mock_latency_ms = mock_latency_ms
        self.mock_boost = mock_boost

        # 解析 API key（兼容 OpenAI）
        if api_key is None:
            api_key = os.environ.get(
                "SILICONFLOW_API_KEY",
                os.environ.get("NIM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            )
        self.api_key = api_key

        # 初始化 OpenAI 客户端（NIM 兼容 OpenAI API）
        if not mock and api_key:
            self._client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            self._client = None

    # ── 协同纠错 ────────────────────────────────────────────────────────────

    def correct(
        self,
        image: np.ndarray,
        local_detections: np.ndarray,
        names: dict[int, str],
    ) -> dict[str, Any]:
        """
        调用云端大模型对本地检测结果进行协同纠错。

        Args:
            image: [H, W, 3] RGB 图像
            local_detections: [N, 6] 本地检测结果 (x1, y1, x2, y2, conf, cls)
            names: 类别名称映射 {class_id: name}

        Returns:
            dict: {
                "detections": [M, 6] 精修后的检测框,
                "latency_ms": 云端推理耗时,
                "raw_text": 大模型原始返回文本（mock 模式下为 None）,
                "success": 是否成功,
            }
        """
        t0 = time.perf_counter()

        if self.mock:
            return self._mock_correct(image, local_detections, names)

        # 构造 prompt
        det_json = detections_to_json(local_detections, names)
        det_text = json.dumps(det_json, ensure_ascii=False, indent=2)

        # 精简 prompt：只给关键信息，减少文本 token
        user_prompt = (
            f"Detected {len(local_detections)} objects by edge model:\n"
            f"{det_text}\n\n"
            f"Verify against the image. Return ONLY a JSON array: "
            f'[{{"class":"car","bbox":[x1,y1,x2,y2],"confidence":0.9}},...]. '
            f"Remove false positives, adjust inaccurate boxes, add missing objects. "
            f"Keep original class names. Return [] if none."
        )

        # 构造消息（OpenAI vision 格式）
        # 上传前压缩图片，避免 prompt token 过多导致 API 超时
        # 硅基流动 Qwen3-VL-8B: 640px 足够，再大 token 暴增、延迟飙升
        image = self._resize_image(image, max_side=640)
        image_b64 = image_to_base64(image, fmt="JPEG")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ]

        # 调用 API
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # 防御性处理：某些模型可能返回空 choices 或空 content
            if not response.choices or not response.choices[0].message:
                print("[CloudClient] API 返回空 choices/message，回退本地结果")
                return {
                    "detections": local_detections,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "raw_text": None,
                    "success": False,
                    "error": "Empty API response",
                }

            raw_text = response.choices[0].message.content or "[]"

            # 调试：打印返回内容前 300 字符，过滤空/无意义内容
            preview = raw_text[:300].replace('\n', ' ')
            if self._should_show_preview(raw_text):
                print(f"[CloudClient] API 返回预览: {preview}...")

            # 解析返回的 JSON
            dets = parse_cloud_response(raw_text, names)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "detections": dets,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "success": True,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[CloudClient] API 调用失败: {e}")
            # 失败时回退到本地结果
            return {
                "detections": local_detections,
                "latency_ms": latency_ms,
                "raw_text": None,
                "success": False,
                "error": str(e),
            }

    # ── Review 模式协同纠错 ─────────────────────────────────────────────────

    def correct_review(
        self,
        image: np.ndarray,
        local_detections: np.ndarray,
        names: dict[int, str],
    ) -> dict[str, Any]:
        """
        协同纠错：VLM 审查边缘检测结果，输出 remove / adjust / add 决策，不修改已有框坐标。
        - remove: 删除误检框
        - adjust: 调整（提升）低置信度框的置信度
        - add: 补充边缘模型漏检的目标
        优势：保持边缘模型高精度坐标；VLM 输出 token 可控；可提升 Precision 和 Recall。
        """
        from .utils import detections_to_review_json, merge_review_decisions, parse_review_decision

        t0 = time.perf_counter()

        if self.mock:
            return self._mock_correct_review(image, local_detections, names)

        det_json = detections_to_review_json(local_detections, names)
        det_text = json.dumps(det_json, ensure_ascii=False, indent=2)

        user_prompt = (
            f"The edge model detected {len(local_detections)} objects in a traffic scene. "
            f"Your PRIMARY goal is to IMPROVE PRECISION by removing false positives. "
            f"Review and CORRECT them against the image:\n\n"
            f"{det_text}\n\n"
            "REVIEW RULES (priority: REMOVE > ADJUST > ADD):\n"
            "1. REMOVE false positives (highest priority — be aggressive):\n"
            "   - Shadows, puddles, or road markings mistaken for vehicles/pedestrians\n"
            "   - Traffic signs, traffic lights, billboards misdetected as vehicles\n"
            "   - Reflections in windows, mirrors, or wet road surfaces\n"
            "   - Building walls, trees, or poles misdetected as pedestrians\n"
            "   - The SAME object detected multiple times with overlapping boxes\n"
            "   - Any box that does NOT actually contain the claimed object class\n"
            "2. ADJUST: Increase confidence for valid detections that look correct but have low confidence (< 0.5).\n"
            "   Format: {\"index\": N, \"confidence\": 0.85}\n"
            "3. ADD missing objects ONLY if you are highly confident (confidence >= 0.7):\n"
            "   - Small or distant vehicles/pedestrians clearly visible\n"
            "   - Partially occluded objects with enough visible features\n"
            "   Format: {\"class\":\"car\",\"bbox\":[x1,y1,x2,y2],\"confidence\":0.8}\n"
            "4. Do NOT remove valid objects that are genuinely present, even if small or distant.\n"
            "   But DO remove anything that is clearly a misdetection.\n\n"
            "OUTPUT FORMAT (strict JSON only, no markdown, no explanation):\n"
            '{\n'
            '  "remove": [index numbers of false positives],\n'
            '  "adjust": [{"index": N, "confidence": 0.85}, ...],\n'
            '  "add": [{"class":"car","bbox":[x1,y1,x2,y2],"confidence":0.8}, ...]\n'
            '}\n'
            "If nothing to change: {\"remove\":[],\"adjust\":[],\"add\":[]}\n"
            "IMPORTANT: The index numbers above are FORMAT EXAMPLES only. "
            "You must determine the ACTUAL indices based on the detection list and image."
        )

        image = self._resize_image(image, max_side=640)
        image_b64 = image_to_base64(image, fmt="JPEG")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if not response.choices or not response.choices[0].message:
                return self._fallback_review(local_detections, t0, "Empty API response")

            raw_text = response.choices[0].message.content or "{}"
            preview = raw_text[:300].replace('\n', ' ')
            if self._should_show_preview(raw_text):
                print(f"[CloudClient] Review 返回预览: {preview}...")

            decision = parse_review_decision(raw_text)
            h, w = image.shape[:2]
            merged, stats = merge_review_decisions(local_detections, decision, names, w, h)
            if stats["removed"] > 0 or stats["added"] > 0:
                print(f"[ReviewStats] 删{stats['removed']} 保{stats['protected']} 调{stats['adjusted']} 补{stats['added']}")

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "detections": merged,
                "decision": decision,
                "stats": stats,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "success": True,
            }

        except Exception as e:
            return self._fallback_review(local_detections, t0, str(e))

    def _fallback_review(self, local_detections: np.ndarray, t0: float, error: str) -> dict[str, Any]:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[CloudClient] Review API 失败: {error}")
        return {
            "detections": local_detections.copy(),
            "decision": {"remove": [], "adjust": [], "add": []},
            "latency_ms": latency_ms,
            "raw_text": None,
            "success": False,
            "error": error,
        }

    def _mock_correct_review(
        self,
        image: np.ndarray,
        local_detections: np.ndarray,
        names: dict[int, str],
    ) -> dict[str, Any]:
        """Mock Review 模式：模拟删除低 conf 框 + 少量补充。"""
        import random
        time.sleep(random.uniform(*self.mock_latency_ms) / 1000.0)

        if len(local_detections) == 0:
            return {
                "detections": local_detections.copy(),
                "decision": {"remove": [], "adjust": [], "add": []},
                "latency_ms": random.uniform(*self.mock_latency_ms),
                "raw_text": None,
                "success": True,
                "mock": True,
            }

        # 模拟删除 conf < 0.3 的框
        remove = [i for i, det in enumerate(local_detections) if det[4] < 0.3]
        keep_mask = np.ones(len(local_detections), dtype=bool)
        for idx in remove:
            keep_mask[idx] = False
        merged = local_detections[keep_mask].copy()

        # 模拟提升剩余框置信度
        if len(merged) > 0:
            merged[:, 4] = np.clip(merged[:, 4] + 0.05, 0.0, 1.0)

        return {
            "detections": merged,
            "decision": {"remove": remove, "adjust": [], "add": []},
            "latency_ms": random.uniform(*self.mock_latency_ms),
            "raw_text": None,
            "success": True,
            "mock": True,
        }

    # ── Mock 模式（旧版兼容）──────────────────────────────────────────────────

    def _mock_correct(
        self,
        image: np.ndarray,
        local_detections: np.ndarray,
        names: dict[int, str],
    ) -> dict[str, Any]:
        """
        模拟云端纠错（不调用真实 API）。
        策略：以一定概率删除低置信度框，以一定概率提升剩余框的置信度。
        """
        import random
        time.sleep(random.uniform(*self.mock_latency_ms) / 1000.0)

        if len(local_detections) == 0:
            return {
                "detections": local_detections.copy(),
                "latency_ms": random.uniform(*self.mock_latency_ms),
                "raw_text": None,
                "success": True,
                "mock": True,
            }

        dets = local_detections.copy()

        # 模拟纠错：删除置信度 < 0.3 的框（模拟大模型过滤误检）
        keep_mask = dets[:, 4] >= 0.3
        dets = dets[keep_mask]

        # 模拟提升剩余框的置信度（模拟大模型验证后更确信）
        if len(dets) > 0:
            boost = 0.05
            dets[:, 4] = np.clip(dets[:, 4] + boost, 0.0, 1.0)

        return {
            "detections": dets,
            "latency_ms": random.uniform(*self.mock_latency_ms),
            "raw_text": None,
            "success": True,
            "mock": True,
        }

    # ── 预览过滤 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _should_show_preview(text: str) -> bool:
        """判断 API 返回预览是否值得显示（过滤空/无意义内容）。"""
        stripped = text.strip()
        return bool(stripped and stripped not in ("[]", "{}", "null", "None", "", "[ ]"))

    # ── 图片压缩 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resize_image(image: np.ndarray, max_side: int = 960) -> np.ndarray:
        """等比例压缩图片，最大边不超过 max_side。"""
        h, w = image.shape[:2]
        if max(h, w) <= max_side:
            return image
        scale = max_side / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        import cv2
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ── 便捷工厂 ────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict, mock: bool | None = None) -> "CloudClient":
        """从配置字典实例化客户端，支持 NIM API / 本地部署 / Mock 三种模式。"""
        if mock is None:
            mock = cfg.get("mode") == "mock" or not cfg.get("use_real_api", False)

        # 根据 mode 选择配置源
        mode = cfg.get("mode", "nim")
        if mode == "local" and "local" in cfg:
            src = cfg["local"]
        elif mode == "nim" and "nim" in cfg:
            src = cfg["nim"]
        else:
            src = cfg  # 兼容旧配置

        # 本地部署默认用更短的 mock 延迟（2-4B VLM 比 11B 快）
        if mode == "local":
            default_min, default_mean, default_std = 300.0, 800.0, 200.0
        else:
            default_min, default_mean, default_std = 200.0, 3000.0, 500.0

        cloud_cfg = cfg.get("cloud_inference_ms", {})

        return cls(
            api_base=src.get("api_base", cfg.get("api_base", "")),
            api_key=src.get("api_key", cfg.get("api_key")),
            model=src.get("model", cfg.get("model", "meta/llama-3.2-11b-vision-instruct")),
            system_prompt=cfg.get("system_prompt", ""),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 512),
            timeout=cfg.get("timeout", 60.0),
            mock=mock,
            mock_latency_ms=(
                cloud_cfg.get("min", default_min),
                cloud_cfg.get("mean", default_mean) + cloud_cfg.get("std", default_std),
            ),
            mock_boost=cfg.get("mock_cloud_accuracy_boost", 0.12),
        )
