"""MobileNetV3 风格搜索空间定义（无 SE），由 config_hw.py 驱动。"""
from skopt.space import Integer

from .config_hw import (
    DEPTH_MAX,
    DEPTH_MIN,
    EXPAND_RATIO_LIST,
    INFER_MODE_LIST,
    KS_LIST,
    NUM_STAGES,
    QUANT_PRECISION_LIST,
    WIDTH_MULTIPLIER_LIST,
)


def _append_index_dim(space, values, name: str) -> bool:
    if len(values) < 1:
        raise ValueError(f"{name} 对应列表不能为空")
    if len(values) > 1:
        space.append(Integer(0, len(values) - 1, name=name))
        return True
    return False


def get_search_space():
    n_ks, n_exp, n_w = len(KS_LIST), len(EXPAND_RATIO_LIST), len(WIDTH_MULTIPLIER_LIST)
    n_inf, n_q = len(INFER_MODE_LIST), len(QUANT_PRECISION_LIST)
    if n_ks < 1 or n_exp < 1 or n_w < 1:
        raise ValueError("KS_LIST、EXPAND_RATIO_LIST、WIDTH_MULTIPLIER_LIST 至少各含一项")
    if n_inf < 1 or n_q < 1:
        raise ValueError("INFER_MODE_LIST、QUANT_PRECISION_LIST 至少各含一项")
    if DEPTH_MIN > DEPTH_MAX:
        raise ValueError("DEPTH_MIN 不能大于 DEPTH_MAX")
    space = []
    _append_index_dim(space, KS_LIST, "ks_idx")
    _append_index_dim(space, EXPAND_RATIO_LIST, "expand_idx")
    _append_index_dim(space, WIDTH_MULTIPLIER_LIST, "width_idx")
    if DEPTH_MIN < DEPTH_MAX:
        for i in range(NUM_STAGES):
            space.append(Integer(DEPTH_MIN, DEPTH_MAX, name=f"depth_s{i}"))
    _append_index_dim(space, INFER_MODE_LIST, "infer_idx")
    _append_index_dim(space, QUANT_PRECISION_LIST, "quant_idx")
    return space


def params_to_arch_config(params):
    idx = 0
    ks = KS_LIST[int(params[idx])] if len(KS_LIST) > 1 else KS_LIST[0]
    idx += 1 if len(KS_LIST) > 1 else 0
    expand = EXPAND_RATIO_LIST[int(params[idx])] if len(EXPAND_RATIO_LIST) > 1 else EXPAND_RATIO_LIST[0]
    idx += 1 if len(EXPAND_RATIO_LIST) > 1 else 0
    width_mult = (
        WIDTH_MULTIPLIER_LIST[int(params[idx])] if len(WIDTH_MULTIPLIER_LIST) > 1 else WIDTH_MULTIPLIER_LIST[0]
    )
    idx += 1 if len(WIDTH_MULTIPLIER_LIST) > 1 else 0
    if DEPTH_MIN < DEPTH_MAX:
        depths = [int(params[idx + i]) for i in range(0, NUM_STAGES)]
        idx += NUM_STAGES
    else:
        depths = [int(DEPTH_MIN) for _ in range(NUM_STAGES)]
    infer_mode = INFER_MODE_LIST[int(params[idx])] if len(INFER_MODE_LIST) > 1 else INFER_MODE_LIST[0]
    idx += 1 if len(INFER_MODE_LIST) > 1 else 0
    quant_precision = (
        QUANT_PRECISION_LIST[int(params[idx])] if len(QUANT_PRECISION_LIST) > 1 else QUANT_PRECISION_LIST[0]
    )
    return {
        "kernel_size": ks,
        "expand_ratio": expand,
        "width_multiplier": width_mult,
        "depths": depths,
        "infer_mode": infer_mode,
        "quant_precision": quant_precision,
    }


def arch_config_to_params(arch_config):
    params = []
    if len(KS_LIST) > 1:
        params.append(KS_LIST.index(arch_config["kernel_size"]))
    if len(EXPAND_RATIO_LIST) > 1:
        params.append(EXPAND_RATIO_LIST.index(arch_config["expand_ratio"]))
    w_mult = arch_config.get("width_multiplier", 1.0)
    if len(WIDTH_MULTIPLIER_LIST) > 1:
        if w_mult in WIDTH_MULTIPLIER_LIST:
            width_idx = WIDTH_MULTIPLIER_LIST.index(w_mult)
        else:
            width_idx = min(
                range(len(WIDTH_MULTIPLIER_LIST)),
                key=lambda i: abs(WIDTH_MULTIPLIER_LIST[i] - w_mult),
            )
        params.append(width_idx)
    depths = arch_config["depths"]
    if len(depths) != NUM_STAGES:
        raise ValueError(
            f"arch_config['depths'] 长度应为 {NUM_STAGES}（与 config_hw.NUM_STAGES 一致），收到 {len(depths)}"
        )
    if DEPTH_MIN < DEPTH_MAX:
        params.extend(list(depths))
    infer_mode = arch_config.get("infer_mode", INFER_MODE_LIST[0])
    infer_idx = INFER_MODE_LIST.index(infer_mode) if infer_mode in INFER_MODE_LIST else 0
    quant_precision = arch_config.get("quant_precision", QUANT_PRECISION_LIST[0])
    quant_idx = QUANT_PRECISION_LIST.index(quant_precision) if quant_precision in QUANT_PRECISION_LIST else 0
    if len(INFER_MODE_LIST) > 1:
        params.append(infer_idx)
    if len(QUANT_PRECISION_LIST) > 1:
        params.append(quant_idx)
    return params
