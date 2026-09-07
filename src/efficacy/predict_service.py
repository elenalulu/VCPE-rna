#!/usr/bin/env python
"""效力模型推理服务：ASO 序列 + 靶基因 (+细胞系) -> 敲低效率预测。

模型：efficacy_v1（ASO Atlas 190,927 gapmer 训练，val spearman 0.50，n=21,989）。
输入：DNA 序列（10-30nt，非 ACGT 字符剔除）；化学默认全 DNA + PS 骨架（gapmer 标准）；
     剂量未知 -> dose_mu + missing flag（训练同款缺失处理）。
细胞系：cell2id 精确匹配（大小写不敏感）；未收录 -> 82 个已训 cell_emb 的均值向量回退。
输出：inhibition_pct（0-95 截断）与 kd（0.20-0.95 截断）。

用法：
  from predict_service import predict_inhibition
  predict_inhibition("TGCATCGTACGTAGCTGATC", "APOC3", cell_line="hepg2")
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.abspath(os.path.join(
    HERE, "..", "..", "results", "efficacy_v1", "ckpt_efficacy_v1.pt"))
DEFAULT_ESM = os.path.abspath(os.path.join(
    HERE, "..", "..", "data", "drive_weights",
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"))

_BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
_pack_cache = {}


def _load(ckpt_fp=None, esm_fp=None):
    ckpt_fp = ckpt_fp or DEFAULT_CKPT
    esm_fp = esm_fp or DEFAULT_ESM
    key = (ckpt_fp, esm_fp)
    if key in _pack_cache:
        return _pack_cache[key]
    from model_efficacy import EfficacyHead
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    tab = torch.load(esm_fp, map_location="cpu", weights_only=False)
    symbols = list(tab.keys())
    esm = torch.stack([v.float() for v in tab.values()])
    gene2row = {s: i for i, s in enumerate(symbols)}
    n_cell = ck["model_state_dict"]["cell_emb.weight"].shape[0]
    m = EfficacyHead(esm, n_cell_lines=n_cell)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n_trained = n_cell - 1                      # 末行未训练（未知回退保留位）
    cell_mean = m.cell_emb.weight[:n_trained].mean(0).detach()
    pack = dict(model=m, gene2row=gene2row, cell2id=ck["cell2id"],
                dose_mu=float(ck["dose_mu"]), cell_mean=cell_mean,
                metrics=ck.get("metrics", {}))
    _pack_cache[key] = pack
    return pack


SUGAR_IDX = {"DNA": 0, "MOE": 1, "cEt": 2, "F": 3, "other": 4}
BB_IDX = {"PO": 0, "PS": 1}
# 训练词表无 LNA 类：LNA 与 cEt 同属约束型 BNA 家族，映射到 cEt（模型已学的最近类）
WING_MAP = {"dna": "DNA", "unmodified": "DNA", "moe": "MOE",
            "lna": "cEt", "cet": "cEt", "f": "F", "mix": "MOE"}


def predict_inhibition(seq, target_gene, cell_line=None,
                       wing_mod=None, wing_len=3,
                       ckpt_fp=None, esm_fp=None):
    """返回 dict；error 键存在表示预测失败（调用方回退手输效率）。"""
    pack = _load(ckpt_fp, esm_fp)
    m = pack["model"]
    seq = "".join(ch for ch in str(seq).upper() if ch in "ACGT")
    if len(seq) < 10 or len(seq) > 30:
        return dict(error=f"序列长度需 10-30nt（当前 {len(seq)}）")
    row = pack["gene2row"].get(target_gene)
    if row is None:
        return dict(error=f"靶基因 {target_gene} 不在 ESM 表覆盖范围")

    L = len(seq)
    base = torch.tensor([[_BASES[ch] for ch in seq]], dtype=torch.long)
    # 化学轴：翼修饰 -> sugar 词表（默认全 DNA + PS）
    wing_eff = WING_MAP.get(str(wing_mod or "dna").lower(), "DNA")
    wl = max(0, min(int(wing_len), (L - 1) // 2))
    sug_idx = SUGAR_IDX[wing_eff]
    sug_row = [sug_idx if (i < wl or i >= L - wl) else SUGAR_IDX["DNA"]
               for i in range(L)]
    sug = torch.tensor([sug_row], dtype=torch.long)
    bb = torch.ones(1, L, dtype=torch.long)         # 全 PS（gapmer 标准）
    pad = torch.zeros(1, L, dtype=torch.bool)
    gr = torch.tensor([row], dtype=torch.long)

    used, cid = "均值嵌入回退（未收录细胞系）", None
    if cell_line:
        c2i = pack["cell2id"]
        hit_key = cell_line if cell_line in c2i else next(
            (k for k in c2i if k.lower() == str(cell_line).lower()), None)
        if hit_key is not None:
            cid, used = c2i[hit_key], hit_key
    cell_vec = (m.cell_emb(torch.tensor([cid], dtype=torch.long))
                if cid is not None else pack["cell_mean"].unsqueeze(0))
    dl = torch.tensor([pack["dose_mu"]], dtype=torch.float32)
    dm = torch.ones(1, dtype=torch.float32)

    with torch.no_grad():
        x = m.in_proj(torch.cat([m.base_emb(base), m.sugar_emb(sug),
                                 m.bb_emb(bb)], dim=-1))
        x = m.encoder(x, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                            x.masked_fill(pad.unsqueeze(-1), -1e4).max(1).values],
                           dim=-1)
        g = m.esm_proj(m.esm_table[gr])
        f = torch.cat([pooled, g, cell_vec, dl.unsqueeze(-1),
                       dm.unsqueeze(-1)], dim=-1)
        pred = m.head(f).squeeze(-1).item()

    inh = float(min(max(pred, 0.0), 95.0))
    kd = float(min(max(pred / 100.0, 0.2), 0.95))
    return dict(inhibition_pct=round(inh, 1), kd=round(kd, 3),
                cell_line_used=used, target=target_gene, raw_pred=round(pred, 2),
                chemistry=dict(wing=wing_eff, wing_len=wl,
                               mapped=("LNA->cEt" if str(wing_mod or "").lower() == "lna"
                                       else None)),
                model="efficacy_v1 (ASO Atlas 190k gapmer, "
                      f"spearman {pack['metrics'].get('spearman', 0):.2f})")


if __name__ == "__main__":
    import json
    print(json.dumps(predict_inhibition(
        "TGCATCGTACGTAGCTGATC", "APOC3", cell_line="hepg2"),
        ensure_ascii=False, indent=2))


# ===== hybrid 路由：统一效力入口（2026-09-07 同裁判对比后拍板） =====
# 已知靶点（XGBoost gene_vocab 93 个）：XGBoost v5（patent-split PCC 0.58，胜 transformer 0.51）
# 新靶点：efficacy_v1 transformer（ESM2 靶基因轴外推，0.29 vs 0.27 略胜且泛化）
_XGB_PROJ = os.environ.get("RNA_ROBOT_HOME", "")  # optional: rna_robot platform root
_XGB_STATE = {"ok": None}


def _xgb_available():
    if _XGB_STATE["ok"] is None:
        try:
            import xgboost  # noqa: F401
            _XGB_STATE["ok"] = os.path.isdir(_XGB_PROJ)
        except Exception:
            _XGB_STATE["ok"] = False
    return _XGB_STATE["ok"]


def _chem_stub(wing_eff, L, wl):
    """构造 train_aso_xgb.extract_chemistry_features 兼容的 chemistry stub（翼修饰语义）。"""
    class _Mod:
        def __init__(self, m, t, pos):
            self.__dict__ = {"modification": m, "type": t, "positions": pos}
    class _Chem:
        def __init__(self, d):
            self.__dict__ = {"__dict__": d}
    mods = []
    if wing_eff != "DNA":
        pos = list(range(1, wl + 1)) + list(range(L - wl + 1, L + 1))
        mods.append(_Mod(wing_eff, "sugar", pos))
    mods.append(_Mod("PS", "backbone", list(range(1, L + 1))))
    return _Chem({"length": L, "modifications": mods})


def predict_kd_hybrid(seq, target_gene, cell_line=None, wing_mod=None, wing_len=3):
    """统一效力入口：按靶基因是否在 XGBoost 词表内路由，返回字段与 predict_inhibition 一致
    （额外带 model_source: 'xgboost_v5' | 'transformer'）。"""
    base = predict_inhibition(seq, target_gene, cell_line=cell_line,
                              wing_mod=wing_mod, wing_len=wing_len)
    base["model_source"] = "transformer"
    if base.get("error") or not _xgb_available():
        return base
    try:
        if _XGB_PROJ not in sys.path:
            sys.path.insert(0, _XGB_PROJ)
        from utils import train_aso_xgb as _tx
        _tx._ensure_aso_model()
        gv = (_tx._aso_meta or {}).get("gene_vocab", {})
        if str(target_gene) not in gv:
            return base
        seq_clean = "".join(ch for ch in str(seq).upper() if ch in "ACGT")
        L = len(seq_clean)
        wing_eff = WING_MAP.get(str(wing_mod or "dna").lower(), "DNA")
        wl = max(0, min(int(wing_len), (L - 1) // 2))
        kd = _tx.predict_aso_kd(seq_clean, cell_line=cell_line or "HeLa",
                            target_gene=str(target_gene),
                            chemistry_obj=_chem_stub(wing_eff, L, wl))
        inh = float(min(max(kd * 100.0, 0.0), 95.0))
        kd_c = float(min(max(kd, 0.2), 0.95))
        return dict(inhibition_pct=round(inh, 1), kd=round(kd_c, 3),
                    cell_line_used=cell_line or "未指定",
                    target=target_gene, raw_pred=round(kd, 3),
                    chemistry=dict(wing=wing_eff, wing_len=wl,
                                   mapped=("LNA->cEt" if str(wing_mod or "").lower() == "lna"
                                           else None)),
                    model_source="xgboost_v5")
    except Exception as e:  # noqa
        print(f"[hybrid] xgb 分支失败回退 transformer: {str(e)[:90]}", flush=True)
        return base
