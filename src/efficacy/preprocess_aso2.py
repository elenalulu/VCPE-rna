"""ASO Atlas 2.0 preprocessing: HELM -> per-position features + official fold join.

HELM examples (v2.0 uses double braces):
  RNA1{{[moe](A)[sp].[moe]([5meC])[sp].d(T)[sp].(A)[sp].[moe](T)}$$$$
Unit = [sugar]?(base-with-optional-brackets)[backbone]?  ; '.'-separated.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas_2"

BODY_RE = re.compile(r"^RNA1\{\{?(.+?)\}+\}\$*$")
UNIT_RE = re.compile(
    r"^(?:\[(?P<sugar>[a-zA-Z0-9\-]+)\])?"
    r"[\(\[]?(?P<core>[a-zA-Z0-9\-]+)[\)\]]?"
    r"(?:\[(?P<bb>[a-zA-Z0-9\-]+)\])?$")
CORE_RE = re.compile(r"^(?P<base>[a-zA-Z]+)(?P<mod>\[[a-zA-Z0-9\-]+\])?$")


def norm_helm(helm):
    """Normalize any brace style to a canonical body string for parsing/joining."""
    h = str(helm).strip()
    i, j = h.find("{"), h.rfind("}")
    if i < 0 or j <= i:
        return None
    return h[i + 1:j].strip("{}")


def parse_helm(helm):
    body = norm_helm(helm)
    if not body:
        return None
    units = [u for u in body.split(".") if u.strip()]
    bases, sugars, backbones = [], [], []
    for i, u in enumerate(units):
        u = u.strip()
        bb, sugar = None, None
        m_tail = re.search(r"\[([a-zA-Z0-9\-]+)\]$", u)      # trailing backbone
        if m_tail:
            bb = m_tail.group(1)
            u = u[:m_tail.start()]
        m_head = re.match(r"^\[([a-zA-Z0-9\-]+)\]", u)        # leading sugar
        if m_head:
            sugar = m_head.group(1)
            u = u[m_head.end():]
        m_core = re.match(r"^([a-zA-Z]?)\(?\[?([a-zA-Z0-9]+)\]?\)?$", u)  # prefix + base
        if not m_core or not m_core.group(2):
            return None
        prefix, base = m_core.group(1), m_core.group(2)
        if sugar is None:
            sugar = prefix if prefix in ("d", "r") else "d"
        bb = bb if bb in ("sp", "po", "other") else ("sp" if bb else
             ("po" if i == len(units) - 1 else "sp"))
        bases.append(base)
        sugars.append(sugar)
        backbones.append(bb)
    if not bases:
        return None
    return bases, sugars, backbones


def main():
    df = pd.read_parquet(DATA / "in_vitro_inhibition.parquet")
    print(f"in_vitro: {df.shape}")

    folds = pd.read_csv(DATA / "folds" / "in_vitro_inhibition_folds.csv.gz")
    folds["norm_key"] = folds["model_input_helm"].map(norm_helm)
    fold_map = (folds.dropna(subset=["norm_key"]).drop_duplicates("norm_key")
                 .set_index("norm_key")[["group", "fold"]])
    df["_norm"] = df["HELM Annotation"].map(norm_helm)
    df = df.join(fold_map, on="_norm")
    print(f"with fold assignment: {df['fold'].notna().sum()}/{len(df)}")

    rows, n_bad = [], 0
    sample_shown = 0
    for helm, inhib, dose, period, cell, sp, treg, lineage in zip(
            df["HELM Annotation"], df["Inhibition_pct"], df["dosage_nm"],
            df["treatment_period_hrs"], df["cell_line"], df["cell_line_species"],
            df["target_RNA"], df["ccle_oncotree_lineage"]):
        parsed = parse_helm(helm)
        if parsed is None:
            n_bad += 1
            if sample_shown < 3:
                print(f"  [unparsed] {str(helm)[:100]}")
                sample_shown += 1
            continue
        bases, sugars, backbones = parsed
        if not (12 <= len(bases) <= 30):
            n_bad += 1
            continue
        rows.append((helm, ",".join(bases), ",".join(sugars), ",".join(backbones),
                     len(bases), float(inhib), dose, period, cell, sp, treg, lineage))
    out = pd.DataFrame(rows, columns=["helm", "base_seq", "sugar_seq", "bb_seq", "L",
                                      "inhibition", "dosage_nm", "period_hrs",
                                      "cell_line", "species", "target_RNA",
                                      "lineage"])
    # fold lookup via NORMALIZED helm key (df's raw-HELM index has dup rows -> join explosion)
    out["_norm"] = out["helm"].map(norm_helm)
    out = out.join(fold_map, on="_norm").drop(columns=["_norm"])
    out["inhibition"] = out["inhibition"].clip(0, 100).astype(np.float32)
    out["dosage_log10"] = np.log10(pd.to_numeric(out["dosage_nm"], errors="coerce")
                                   .astype(np.float32) + 1.0)
    out["period_log10"] = np.log10(pd.to_numeric(out["period_hrs"], errors="coerce")
                                   .astype(np.float32) + 1.0)
    out.to_parquet(DATA / "in_vitro_clean.parquet", index=False)
    print(f"✅ wrote in_vitro_clean.parquet: {out.shape} (parse failures: {n_bad})")
    print(f"with fold: {out['fold'].notna().sum()}")
    print("sugar vocab:", sorted({s for ss in out["sugar_seq"] for s in ss.split(",")}))
    print("base vocab:", sorted({b for bs in out["base_seq"] for b in bs.split(",")}))
    print("backbone vocab:", sorted({b for bs in out["bb_seq"] for b in bs.split(",")}))


if __name__ == "__main__":
    main()
