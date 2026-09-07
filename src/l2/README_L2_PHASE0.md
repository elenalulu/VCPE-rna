# L2 Phase 0 运行手册（数据准备，全部本机）

前置：`src/l2/` 三个脚本；产物目录建议 `D:/data/l2/`（磁盘预算 ~50GB）。

## 运行顺序

```bash
cd D:/WorkBuddy/alphafold-web/VCPE/VCPE_rna/src/l2

# 1. 下载（10.92GB，断点续传；--list-only 可先看清单）
python download_rnacentral.py --out-dir D:/data/l2 --list-only
python download_rnacentral.py --out-dir D:/data/l2

# 2. 过滤 + 精确去重（20-500nt、N<=10%、归一化 T->U；流式，内存 ~1GB）
python filter_dedup.py --fasta D:/data/l2/rnacentral_active.fasta.gz \
  --out D:/data/l2/l2_filtered.fasta

# 3. Token 化存储（扁平 uint16 + offsets，无 padding；train/val 99.5/0.5）
python tokenize_store.py --fasta D:/data/l2/l2_filtered.fasta --out-dir D:/data/l2
```

## 产物清单

| 文件 | 用途 |
|---|---|
| `rnacentral_active.fasta.gz` | 原始语料（10.92GB，可删） |
| `l2_filtered.fasta` | 过滤+去重后的训练语料（~5-15M 条） |
| `l2_tokens.u16` + `l2_offsets.npy` | token 化存储（L2-1 预训练直接读） |
| `l2_train_idx.npy` / `l2_val_idx.npy` | 行索引切分 |
| `*.stats.json` / `tokenize_stats.json` | 各阶段统计（回传诊断用） |

## 时间/空间预估

| 步骤 | 时间 | 磁盘增量 |
|---|---|---|
| 下载 | 1-3 小时（带宽决定） | +11GB |
| 过滤去重 | 1-2 小时（流式单进程） | +8-12GB |
| Token 化 | 1-2 小时 | +5-8GB |

## 传 GPU 机

`l2_tokens.u16 + l2_offsets.npy + l2_train_idx.npy + l2_val_idx.npy + tokenize_stats.json`
（~5-8GB，L2-1 预训练的唯一输入；原始 fasta 不用传）
