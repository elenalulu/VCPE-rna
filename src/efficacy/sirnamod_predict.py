# -*- coding: utf-8 -*-
"""siRNAmod 修饰感知预测（平台集成入口）。

用法 ①（效力路由）：predict_modified_efficiency(sense, antisense, mod_pattern) → 0-1 校正后效率
用法 ②（独立工具）：predict_inhibition(sense, antisense, sense_mods, antisense_mods) → 0-100 抑制率
"""
import joblib
import os
import re

import numpy as np

_MODEL = None
_FEAT_NAMES = None
_MOD_ORDER = ['unmod', 'fl2r', 's4r', 'lna', 'hna', 'una', 'ome', 'dna']
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "data", "drive_weights", "sirnamod_xgb_v1.joblib")

# 常见修饰模式预设 → (sense_mods_fn, antisense_mods_fn)
MOD_PATTERNS = {
    "unmodified": lambda seq: (['unmod'] * len(seq), ['unmod'] * len(seq)),
    "2F_alternating": lambda seq: (['fl2r' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                    ['unmod'] * len(seq)),
    "2OMe_alternating": lambda seq: (['ome' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                      ['unmod'] * len(seq)),
    "2F2OMe_standard": lambda seq: (['fl2r' if i % 2 == 0 else 'ome' for i in range(len(seq))],
                                     ['fl2r' if i % 2 == 0 else 'ome' for i in range(len(seq))]),
    "full_2F": lambda seq: (['fl2r'] * len(seq), ['fl2r'] * len(seq)),
    "LNA_alternating": lambda seq: (['lna' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                     ['unmod'] * len(seq)),
    "s4r_sense": lambda seq: (['s4r'] * len(seq), ['unmod'] * len(seq)),
}


def _ensure_model():
    global _MODEL, _FEAT_NAMES
    if _MODEL is not None:
        return
    data = joblib.load(_MODEL_PATH)
    _MODEL = data['model']
    _FEAT_NAMES = data['feat_names']


def build_features(sense, antisense, max_len=25):
    """与 sirnamod_model.build_features 完全一致的特征构建。

    Args:
        sense: list of (base, mod) tuples
        antisense: list of (base, mod) tuples
    Returns:
        dict of features
    """
    feats = {}
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        mod_counts = {}
        for _, m in nts:
            mod_counts[m] = mod_counts.get(m, 0) + 1
        for m in _MOD_ORDER:
            feats[f'{strand_name}_n_{m}'] = mod_counts.get(m, 0)
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for m in _MOD_ORDER:
                feats[f'{strand_name}{pos}_{m}'] = 1 if mod == m else 0
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for b in 'ACGU':
                feats[f'{strand_name}{pos}_{b}'] = 1 if base == b else 0
    if len(antisense) >= 12:
        for m in _MOD_ORDER:
            seed_count = sum(1 for pos, (b, mo) in enumerate(antisense[1:8]) if mo == m)
            feats[f'Aseed_{m}'] = seed_count
            cleav_count = sum(1 for pos, (b, mo) in enumerate(antisense[9:12]) if mo == m)
            feats[f'Aclev_{m}'] = cleav_count
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        gc = sum(1 for b, _ in nts if b in 'GC') / max(len(nts), 1)
        feats[f'{strand_name}_GC'] = gc
    return feats


def predict_inhibition(sense_seq, antisense_seq, sense_mods=None, antisense_mods=None):
    """预测修饰 siRNA 抑制率 %。

    Args:
        sense_seq: str 引导链序列（19-25 nt）
        antisense_seq: str 反义链序列
        sense_mods: list[str] per-position 修饰类型（None=全 unmod）
        antisense_mods: list[str] 同上

    Returns:
        float, 预测抑制率 %（0-100，clip 到 [0, 100]）
    """
    _ensure_model()
    sense_nts = [(b, (sense_mods[i] if sense_mods and i < len(sense_mods) else 'unmod'))
                 for i, b in enumerate(sense_seq.upper().replace('T', 'U'))]
    antisense_nts = [(b, (antisense_mods[i] if antisense_mods and i < len(antisense_mods) else 'unmod'))
                     for i, b in enumerate(antisense_seq.upper().replace('T', 'U'))]
    feats = build_features(sense_nts, antisense_nts)
    x = np.array([[feats.get(k, 0) for k in _FEAT_NAMES]], dtype=np.float32)
    pred = float(_MODEL.predict(x)[0])
    return max(0.0, min(100.0, pred))


def predict_modified_efficiency(sense_seq, antisense_seq, mod_pattern="unmodified"):
    """效力路由集成入口：修饰模式 → 校正后敲低效率（0-1）。

    Args:
        sense_seq: str 引导链序列
        antisense_seq: str 反义链序列
        mod_pattern: str 修饰模式名（MOD_PATTERNS 的 key）或 None

    Returns:
        (efficiency_0_1, sirnamod_inhibition_pct, mod_label)
        - efficiency: 0-1，sirnamod 预测抑制率 / 100
        - sirnamod_inhibition_pct: 原始预测抑制率 %
        - mod_label: 修饰模式描述
    """
    if mod_pattern and mod_pattern in MOD_PATTERNS:
        sense_mods, antisense_mods = MOD_PATTERNS[mod_pattern](sense_seq)
        label = mod_pattern
    else:
        sense_mods = antisense_mods = None
        label = "unmodified"
    inh = predict_inhibition(sense_seq, antisense_seq, sense_mods, antisense_mods)
    eff = inh / 100.0
    return eff, inh, label


def get_mod_pattern_names():
    """返回支持的修饰模式名列表。"""
    return list(MOD_PATTERNS.keys())
