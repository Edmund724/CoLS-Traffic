#!/usr/bin/env python3
"""
Teacher Cache Generator for VLM-based Knowledge Distillation.

Uses NVIDIA NIM API (Llama-3.2-Vision-11B) to generate two types of
teacher outputs for each image in DAIR-V2X:

1. Scene Complexity Analysis -> for Gating Network training (Phase 4)
   and hard-sample mining (Phase 2).

2. Structured Detection Pseudo-labels -> auxiliary supervision for
   student detector distillation (Phase 2).

Outputs are saved as JSONL for easy streaming during training.
Supports resume from interruption.
"""

import os
import sys
import json
import base64
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Try common env var names for NVIDIA NIM API key
API_KEY = (
    os.environ.get("NVIDIA_API_KEY")
    or os.environ.get("NIM_API_KEY")
    or os.environ.get("NGC_API_KEY")
)

# NVIDIA NIM API endpoint (OpenAI-compatible)
BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

MODEL_NAME = os.environ.get("NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")

# DAIR-V2X class names (must match dataset yaml)
CLASS_NAMES = [
    "car", "truck", "van", "bus", "pedestrian",
    "cyclist", "tricyclist", "motorcyclist", "barrowlist", "trafficcone",
]

# Rate limit: 40 RPM -> 1 request every 1.5 seconds
MAX_WORKERS = int(os.environ.get("CACHE_MAX_WORKERS", "1"))
REQUEST_DELAY = float(os.environ.get("CACHE_REQUEST_DELAY", "1.5"))
MAX_RETRIES = int(os.environ.get("CACHE_MAX_RETRIES", "3"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPLEXITY_SYSTEM_PROMPT = (
    "You are an expert traffic scene analyst. "
    "You MUST respond with ONLY a single valid JSON object. "
    "Do not include markdown formatting, explanations, or any text outside the JSON."
)

COMPLEXITY_USER_PROMPT = (
    "Analyze this traffic scene image. Consider: weather conditions, lighting "
    "(day/night/dusk), target density, occlusion level, image blur/quality, "
    "and unusual scenarios.\n\n"
    "Output ONLY a JSON object with exactly these keys:\n"
    "- complexity_score: number from 0.0 (very easy) to 1.0 (extremely difficult)\n"
    "- weather: one of [clear, rain, snow, fog, unknown]\n"
    "- lighting: one of [day, night, dusk_dawn, backlit, unknown]\n"
    "- target_density: one of [sparse, moderate, dense, extremely_dense]\n"
    "- occlusion_level: one of [none, light, moderate, heavy]\n"
    "- image_quality: one of [good, moderate, poor, very_poor]\n"
    "- reasoning: string, 1-2 sentences explaining the score\n\n"
    "No markdown, no extra text, only the JSON object."
)


DETECTION_SYSTEM_PROMPT = (
    "You are an expert traffic object detector. "
    "You MUST respond with ONLY a single valid JSON object. "
    "Do not include markdown formatting, explanations, or any text outside the JSON."
)

_DETECTION_CLASSES = ", ".join(CLASS_NAMES)

DETECTION_USER_PROMPT = (
    f"Detect all traffic objects in this image. Valid classes: {_DETECTION_CLASSES}.\n\n"
    "For each object, provide its class name and bounding box in normalized "
    "coordinates [x_min, y_min, x_max, y_max] where 0.0 and 1.0 correspond to "
    "the image edges. Also provide your confidence (0.0-1.0).\n\n"
    "If you are unsure about an object, either omit it or assign a low confidence.\n\n"
    "Output ONLY a JSON object with exactly these keys:\n"
    "- objects: array of {class, bbox [x_min, y_min, x_max, y_max], confidence}\n"
    "- scene_description: string, brief 1-sentence scene summary\n\n"
    "No markdown, no extra text, only the JSON object."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComplexityRecord:
    image_path: str
    image_id: str
    complexity_score: float
    weather: str
    lighting: str
    target_density: str
    occlusion_level: str
    image_quality: str
    reasoning: str
    model: str
    timestamp: str


@dataclass
class DetectionRecord:
    image_path: str
    image_id: str
    objects: list
    scene_description: str
    model: str
    timestamp: str


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    """Encode image to base64 data URL."""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Optional[dict]:
    """Extract and parse the first JSON object from text."""
    # Try to find a JSON object bounded by braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def make_request(
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> Optional[dict]:
    """Call NVIDIA NIM API and extract JSON from response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)
    image_b64_url = encode_image(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_b64_url}},
            ],
        },
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,  # low temp for consistency
            )
            content = response.choices[0].message.content
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
            # JSON extraction failed, retry
            print(f"  [Retry {attempt+1}/{MAX_RETRIES}] JSON extraction failed from response")
            time.sleep(2 ** attempt)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Retry {attempt+1}/{MAX_RETRIES}] {e} -> wait {wait}s")
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Core caching logic
# ---------------------------------------------------------------------------

def load_done_set(jsonl_path: str) -> set:
    """Load set of already-processed image_ids from existing JSONL."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("image_id", ""))
            except json.JSONDecodeError:
                continue
    return done


def get_image_id(image_path: str) -> str:
    """Generate a stable image identifier."""
    return str(Path(image_path))


def process_image_complexity(image_path: str, out_path: str) -> bool:
    """Process a single image for scene complexity. Returns True on success."""
    image_id = get_image_id(image_path)
    result = make_request(
        image_path=image_path,
        system_prompt=COMPLEXITY_SYSTEM_PROMPT,
        user_prompt=COMPLEXITY_USER_PROMPT,
        max_tokens=512,
    )
    if result is None:
        return False

    # Validate required fields with defaults
    record = ComplexityRecord(
        image_path=image_path,
        image_id=image_id,
        complexity_score=float(result.get("complexity_score", 0.5)),
        weather=result.get("weather", "unknown"),
        lighting=result.get("lighting", "unknown"),
        target_density=result.get("target_density", "moderate"),
        occlusion_level=result.get("occlusion_level", "light"),
        image_quality=result.get("image_quality", "good"),
        reasoning=result.get("reasoning", ""),
        model=MODEL_NAME,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return True


def process_image_detection(image_path: str, out_path: str) -> bool:
    """Process a single image for structured detection. Returns True on success."""
    image_id = get_image_id(image_path)
    result = make_request(
        image_path=image_path,
        system_prompt=DETECTION_SYSTEM_PROMPT,
        user_prompt=DETECTION_USER_PROMPT,
        max_tokens=2048,
    )
    if result is None:
        return False

    record = DetectionRecord(
        image_path=image_path,
        image_id=image_id,
        objects=result.get("objects", []),
        scene_description=result.get("scene_description", ""),
        model=MODEL_NAME,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def get_image_list(data_yaml: str, split: str = "train") -> list:
    """Load image paths from Ultralytics YAML dataset config."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["path"]).resolve()
    img_dir = data_root / cfg[split]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for ext in exts:
        images.extend(img_dir.glob(f"*{ext}"))
        images.extend(img_dir.glob(f"*{ext.upper()}"))

    return sorted([str(p.resolve()) for p in images])


def run_cache(
    data_yaml: str,
    output_dir: str,
    split: str = "train",
    mode: str = "both",  # "complexity", "detection", or "both"
    limit: Optional[int] = None,
):
    """Main entry point for caching teacher outputs."""
    if API_KEY is None:
        print("ERROR: No API key found. Set one of:")
        print("  NVIDIA_API_KEY, NIM_API_KEY, or NGC_API_KEY")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    images = get_image_list(data_yaml, split)
    if limit is not None:
        images = images[:limit]
    print(f"[{split}] Found {len(images)} images in {data_yaml}")

    tasks = []
    if mode in ("complexity", "both"):
        tasks.append("complexity")
    if mode in ("detection", "both"):
        tasks.append("detection")

    for task in tasks:
        out_path = os.path.join(output_dir, f"teacher_{task}_{split}.jsonl")
        done_set = load_done_set(out_path)
        todo = [p for p in images if get_image_id(p) not in done_set]

        print(f"\n[{task}] Already cached: {len(done_set)}, Remaining: {len(todo)}")
        print(f"Output: {out_path}")

        if not todo:
            print(f"[{task}] All done!")
            continue

        success = 0
        fail = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_path = {}
            for img_path in todo:
                if task == "complexity":
                    fn = process_image_complexity
                else:
                    fn = process_image_detection
                fut = executor.submit(fn, img_path, out_path)
                future_to_path[fut] = img_path
                time.sleep(REQUEST_DELAY)

            for i, fut in enumerate(as_completed(future_to_path)):
                img_path = future_to_path[fut]
                try:
                    ok = fut.result()
                    if ok:
                        success += 1
                    else:
                        fail += 1
                        print(f"  FAIL: {os.path.basename(img_path)}")
                except Exception as e:
                    fail += 1
                    print(f"  EXCEPTION: {os.path.basename(img_path)} -> {e}")

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(todo)} (success={success}, fail={fail})")

        print(f"[{task}] Finished: success={success}, fail={fail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache VLM teacher outputs via NVIDIA NIM API")
    parser.add_argument("--data", default="/mnt/d/gra/datasets/dair_v2x_yolo/dair_v2x.yaml",
                        help="Path to Ultralytics YAML dataset config")
    parser.add_argument("--output", default="/mnt/d/gra/datasets/dair_v2x_yolo/teacher_cache",
                        help="Output directory for cached JSONL files")
    parser.add_argument("--split", default="train", choices=["train", "val", "both"],
                        help="Which split to process")
    parser.add_argument("--mode", default="both", choices=["complexity", "detection", "both"],
                        help="Which teacher output to generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N images (for testing)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Max concurrent API requests")

    args = parser.parse_args()
    MAX_WORKERS = args.workers

    if args.split == "both":
        for sp in ["train", "val"]:
            run_cache(args.data, args.output, sp, args.mode, args.limit)
    else:
        run_cache(args.data, args.output, args.split, args.mode, args.limit)
