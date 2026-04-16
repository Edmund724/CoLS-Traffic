"""MobileNetV3 风格搜索空间定义（无 SE），由 config.py 驱动。"""
from skopt.space import Integer

from .config import (
    DEPTH_MAX,
    DEPTH_MIN,
    EXPAND_RATIO_LIST,
    KS_LIST,
    NUM_STAGES,
    WIDTH_MULTIPLIER_LIST,
)


def get_search_space():
    """返回 skopt 搜索空间；各维度上界由 config 中列表长度与 DEPTH_* / NUM_STAGES 决定。"""
    n_ks, n_exp, n_w = len(KS_LIST), len(EXPAND_RATIO_LIST), len(WIDTH_MULTIPLIER_LIST)
    if n_ks < 1 or n_exp < 1 or n_w < 1:
        raise ValueError("KS_LIST、EXPAND_RATIO_LIST、WIDTH_MULTIPLIER_LIST 至少各含一项")
    if DEPTH_MIN > DEPTH_MAX:
        raise ValueError("DEPTH_MIN 不能大于 DEPTH_MAX")
    space = [
        Integer(0, n_ks - 1, name="ks_idx"),
        Integer(0, n_exp - 1, name="expand_idx"),
        Integer(0, n_w - 1, name="width_idx"),
    ]
    for i in range(NUM_STAGES):
        space.append(Integer(DEPTH_MIN, DEPTH_MAX, name=f"depth_s{i}"))
    return space


def params_to_arch_config(params):
    """将 skopt 返回的 params 转为模型构建所需的 arch_config。"""
    ks = KS_LIST[int(params[0])]
    expand = EXPAND_RATIO_LIST[int(params[1])]
    width_mult = WIDTH_MULTIPLIER_LIST[int(params[2])]
    depths = [int(params[i]) for i in range(3, 3 + NUM_STAGES)]
    return {
        "kernel_size": ks,
        "expand_ratio": expand,
        "width_multiplier": width_mult,
        "depths": depths,
    }


def arch_config_to_params(arch_config):
    """将 arch_config 转为 skopt params，用于断点续跑时从已保存模型恢复。"""
    ks_idx = KS_LIST.index(arch_config["kernel_size"])
    expand_idx = EXPAND_RATIO_LIST.index(arch_config["expand_ratio"])
    w_mult = arch_config.get("width_multiplier", 1.0)
    if w_mult in WIDTH_MULTIPLIER_LIST:
        width_idx = WIDTH_MULTIPLIER_LIST.index(w_mult)
    else:
        width_idx = min(
            range(len(WIDTH_MULTIPLIER_LIST)),
            key=lambda i: abs(WIDTH_MULTIPLIER_LIST[i] - w_mult),
        )
    depths = arch_config["depths"]
    if len(depths) != NUM_STAGES:
        raise ValueError(
            f"arch_config['depths'] 长度应为 {NUM_STAGES}（与 config.NUM_STAGES 一致），收到 {len(depths)}"
        )
    return [ks_idx, expand_idx, width_idx] + list(depths)
