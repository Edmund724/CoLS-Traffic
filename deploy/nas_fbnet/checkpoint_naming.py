"""Checkpoint 文件名中的部署/精度后缀：与 export、TensorRT 解析约定一致。"""
import os
from typing import Optional, Tuple

# 右锚点（扩展名前）
_QUANT_SUFFIXES = ("_fp16", "_int8")
_INFER_MARKER = "_infer_"


def infer_filename_suffix(infer_mode: str, quant_precision: str) -> str:
    """返回 ``_infer_{mode}_{precision}``（不含 .pth）。precision 须为 fp16 或 int8。"""
    if quant_precision not in ("fp16", "int8"):
        raise ValueError(f"quant_precision 须为 fp16 或 int8，收到 {quant_precision!r}")
    return f"{_INFER_MARKER}{infer_mode}_{quant_precision}"


def parse_infer_from_filename(path_or_basename: str) -> Optional[Tuple[str, str]]:
    """从文件名解析 (infer_mode, quant_precision)。

    约定：左锚 ``_infer_``；右锚 ``_fp16`` 或 ``_int8``（紧邻 ``.pth`` 之前）。
    ``_infer_`` 与 ``_fp16``/``_int8`` 之间的整段为模式串（可含连字符，如 only-gpu）。
    """
    base = os.path.basename(path_or_basename)
    if not base.endswith(".pth"):
        return None
    stem = base[: -len(".pth")]
    for prec in ("fp16", "int8"):
        suf = f"_{prec}"
        if not stem.endswith(suf):
            continue
        prefix = stem[: -len(suf)]
        pos = prefix.rfind(_INFER_MARKER)
        if pos < 0:
            continue
        mode = prefix[pos + len(_INFER_MARKER) :]
        if mode:
            return mode, prec
    return None
