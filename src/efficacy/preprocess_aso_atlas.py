"""ASO Atlas preprocessing: pickled repo-dependent DataFrame -> clean, self-contained parquet.

Parses the per-position Modification objects into flat per-position feature strings so
downstream training never needs the source repo:
  sugar_seq    e.g. "MOE,MOE,cEt,DNA,...,cEt"     (len == aso length, 1-based positions)
  backbone_seq e.g. "PS,PS,PO,..."
Adds:
  inhibition_clipped  (clip to [0, 130]; raw range -786..224 is measurement artifact)
  dosage_log10        (log10 nM + 1; NaN kept, training adds missing-flag)
Splits by custom_id (source patent table) are done at training time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas"

sys.path.insert(0, str(DATA))  # pickle references module "src.*" (repo layout)

SUGARS = ["DNA", "MOE", "cEt", "F", "other"]
BACKBONES = ["PO", "PS"]


def parse_chemistry(chem_obj, seq_len):
    """Modification objects -> (sugar list, backbone list) of length seq_len."""
    sugar = ["DNA"] * seq_len
    backbone = ["PO"] * seq_len
    if chem_obj is None:
        return sugar, backbone
    mods = getattr(chem_obj, "modifications", None) or []
    for m in mods:
        kind = getattr(m, "type", "")
        name = getattr(m, "modification", "other")
        positions = getattr(m, "positions", []) or []
        for p in positions:
            i = int(p) - 1
            if 0 <= i < seq_len:
                if kind == "sugar":
                    sugar[i] = name if name in SUGARS else "other"
                elif kind == "backbone":
                    backbone[i] = name if name in BACKBONES else "PO"
    return sugar, backbone


def main():
    df = pd.read_pickle(DATA / "aso_atlas.pkl")
    print(f"raw: {df.shape}")

    keep = df.dropna(subset=["aso_sequence_5_to_3", "inhibition_percent"]).copy()
    keep["L"] = keep["aso_sequence_5_to_3"].str.len()
    keep = keep[(keep["L"] >= 12) & (keep["L"] <= 30)]
    print(f"after length filter: {keep.shape}")

    sugars, backbones, ok_chem = [], [], 0
    for chem, seq in zip(keep["chemistry"], keep["aso_sequence_5_to_3"]):
        s, b = parse_chemistry(chem, len(seq))
        sugars.append(",".join(s))
        backbones.append(",".join(b))
        ok_chem += int(chem is not None)
    keep["sugar_seq"] = sugars
    keep["backbone_seq"] = backbones
    print(f"chemistry parsed: {ok_chem}/{len(keep)} with annotation")

    keep["inhibition_clipped"] = keep["inhibition_percent"].clip(0, 130).astype(np.float32)
    dose_num = pd.to_numeric(keep["dosage"], errors="coerce").astype(np.float32)
    keep["dosage_log10"] = np.log10(dose_num + 1.0)  # NaN kept; training adds missing flag
    keep["target_gene"] = keep["target_gene"].fillna("UNKNOWN")
    keep["cell_line"] = keep["cell_line"].fillna("UNKNOWN")

    out_cols = ["aso_sequence_5_to_3", "sugar_seq", "backbone_seq", "L",
                "inhibition_clipped", "target_gene", "cell_line",
                "cell_line_species", "dosage_log10", "transfection_method",
                "custom_id", "steric_blocking"]
    out = keep[out_cols]
    out.to_parquet(DATA / "aso_atlas_clean.parquet", index=False)
    print(f"✅ wrote {DATA / 'aso_atlas_clean.parquet'}: {out.shape}")
    print(out["inhibition_clipped"].describe().to_string())
    print("gene-heldout candidates (genes with >=300 rows):",
          int((out["target_gene"].value_counts() >= 300).sum()))


if __name__ == "__main__":
    main()
