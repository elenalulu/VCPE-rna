# -*- coding: utf-8 -*-
"""siRNAmod (Martinelli 2023, 907 修饰 siRNA) 效力模型。

任务：HELM 修饰模式 + 序列 → 抑制率 %（0-100）
方法：XGBoost 5-fold CV（CPU <1min）
特征：
  L1 全局：每条链每种修饰计数（6×2=12）
  L2 位置：每位置修饰 one-hot（~21×2×6=252）+ 序列 one-hot（~21×2×4=168）
  L3 交叉：seed 区（antisense 2-8）修饰指示 / 切割区（antisense 10-11）修饰指示
评估：5-fold CV Pearson/Spearman + RMSE
"""
import gzip
import csv
import re
import collections
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "oligogym")
MOD_ORDER = ['unmod', 'fl2r', 's4r', 'lna', 'hna', 'una', 'ome', 'dna']


def parse_helm_strand(units):
    """Parse a list of HELM unit tokens into (base, mod) tuples, excluding phosphates."""
    nts = []
    for unit in units:
        if unit.rstrip('.') in ('p', 'p.'):
            continue
        base_m = re.search(r'\(([A-Z])\)', unit)
        base = base_m.group(1) if base_m else '?'
        # detect modification
        bracket_mods = re.findall(r'\[([^\]]+)\]', unit)
        if bracket_mods:
            mod = bracket_mods[0]
        elif unit.startswith('m('):
            mod = 'ome'
        elif unit.startswith('r(') or unit.startswith('[r]'):
            mod = 'unmod'
        elif unit.startswith('d('):
            mod = 'dna'
        else:
            mod = 'unmod'
        nts.append((base, mod))
    return nts


def parse_helm(helm):
    """Parse full HELM string -> (sense_nts, antisense_nts)."""
    strands_raw = helm.split('|')
    parsed = []
    for s in strands_raw:
        m = re.search(r'\{(.+)\}', s)
        if not m:
            parsed.append([])
            continue
        content = m.group(1)
        units = re.findall(r'(\[?[a-z0-9]+\]?\([A-Z]\)|p\b\.?)', content)
        parsed.append(parse_helm_strand(units))
    # strand 0 = sense, strand 1 = antisense
    sense = parsed[0] if len(parsed) > 0 else []
    antisense = parsed[1] if len(parsed) > 1 else []
    return sense, antisense


def build_features(sense, antisense, max_len=25):
    """Build feature vector for one siRNA."""
    feats = {}
    # L1: global mod counts per strand
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        mod_counts = collections.Counter(m for _, m in nts)
        for m in MOD_ORDER:
            feats[f'{strand_name}_n_{m}'] = mod_counts.get(m, 0)
    # L2: positional mod one-hot
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for m in MOD_ORDER:
                feats[f'{strand_name}{pos}_{m}'] = 1 if mod == m else 0
    # sequence one-hot
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for b in 'ACGU':
                feats[f'{strand_name}{pos}_{b}'] = 1 if base == b else 0
    # L3: functional region indicators
    # seed region = antisense positions 1-7 (0-indexed), cleavage site = 9-11
    if len(antisense) >= 12:
        for m in MOD_ORDER:
            seed_count = sum(1 for pos, (b, mo) in enumerate(antisense[1:8]) if mo == m)
            feats[f'Aseed_{m}'] = seed_count
            cleav_count = sum(1 for pos, (b, mo) in enumerate(antisense[9:12]) if mo == m)
            feats[f'Aclev_{m}'] = cleav_count
    # GC content
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        gc = sum(1 for b, _ in nts if b in 'GC') / max(len(nts), 1)
        feats[f'{strand_name}_GC'] = gc
    return feats


def load_data():
    rows = []
    fp = os.path.join(DATA, 'martinelli_2023_1.csv.gz')
    with gzip.open(fp, 'rt', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    X_dicts, y_vals = [], []
    for r in rows:
        sense, antisense = parse_helm(r['x'])
        feats = build_features(sense, antisense)
        X_dicts.append(feats)
        y_vals.append(float(r['y']))
    # 统一特征空间
    all_keys = sorted(set(k for d in X_dicts for k in d))
    X = np.array([[d.get(k, 0) for k in all_keys] for d in X_dicts], dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y, all_keys


def evaluate(X, y, n_folds=5, seed=42):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    import xgboost as xgb

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pearsons, spearmans, rmses = [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
            random_state=seed + fold, n_jobs=-1, verbosity=0)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        pearsons.append(pearsonr(y[te], pred)[0])
        spearmans.append(spearmanr(y[te], pred)[0])
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
    return {
        'pearson_mean': np.mean(pearsons), 'pearson_std': np.std(pearsons),
        'spearman_mean': np.mean(spearmans), 'spearman_std': np.std(spearmans),
        'rmse_mean': np.mean(rmses),
        'fold_pearsons': [round(p, 3) for p in pearsons],
    }


if __name__ == '__main__':
    X, y, feat_names = load_data()
    print(f'X: {X.shape}, y: {y.shape} | y range: {y.min():.1f} - {y.max():.1f} | mean: {y.mean():.1f}')
    print(f'features: {len(feat_names)}')

    res = evaluate(X, y)
    print('\n=== 5-fold CV ===')
    print(f"Pearson:  {res['pearson_mean']:.3f} ± {res['pearson_std']:.3f}")
    print(f"Spearman: {res['spearman_mean']:.3f} ± {res['spearman_std']:.3f}")
    print(f"RMSE:     {res['rmse_mean']:.2f}")
    print(f"Folds:    {res['fold_pearsons']}")

    # 修饰效应分析：各修饰类型的平均抑制率差异
    print('\n=== 修饰效应 ===')
    for m in ['fl2r', 's4r', 'lna', 'hna', 'una']:
        idx_m = [i for i, fn in enumerate(feat_names) if fn == f'sense_n_{m}']
        if not idx_m:
            continue
        j = idx_m[0]
        has = X[:, j] > 0
        if has.sum() > 5:
            print(f'  sense {m}: n={int(has.sum())} | y={y[has].mean():.1f} vs 无={y[~has].mean():.1f} | Δ={y[has].mean()-y[~has].mean():+.1f}')
