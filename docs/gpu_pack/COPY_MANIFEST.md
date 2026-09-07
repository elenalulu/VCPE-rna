# GPU 传输清单（本机 → GPU 机）

> 总量 ≈ 13.6 GB。传输方式由你定（scp / rsync / 网盘均可）；GPU 侧根目录统一 `~/map_rna/`。

## ① 代码与小文件（先传，秒级）

| 本机路径（D:\WorkBuddy\alphafold-web\AIDSi\ 下） | → GPU 机路径 | 说明 |
|---|---|---|
| `VCPE_rna/src/maprna_p1/`（3 个 py，共 28KB） | `~/map_rna/src/` | P1 训练脚本 |
| `VCPE_rna/gpu_pack/`（TASK_P1_001.md + CODEBUDDY_BRIEF.md + README_P1.md） | `~/map_rna/gpu_pack/` | 任务卡 + 简报（喂给 CodeBuddy） |
| `VCPE_rna/PLAN.md` | `~/map_rna/PLAN.md` | 项目规划（CodeBuddy 背景参考） |
| `MAP-KG-main/`（**排除 `.git.bak`**，110MB） | `~/map_rna/MAP-KG-main/` | 已打 10 个补丁的上游 repo（必须完整，含 git 历史） |

排除项：`MAP-KG-main/.git.bak`（23MB，仅本机留档）可传可不传。

## ② 权重（7.8 GB）

| 本机路径（...\AIDSi\VCPE_rna\data\drive_weights\） | → GPU 机路径 |
|---|---|
| `epoch_3.pt`（4.2 GB） | `~/map_rna/weights/` |
| `se600m.safetensors`（2.4 GB） | `~/map_rna/weights/` |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`（411 MB） | `~/map_rna/weights/` |
| `mapkg_encoder_v3.pt`（1.1 GB） | `~/map_rna/weights/` |

## ③ 训练数据（5.4 GB，三份 GEARS h5ad）

| 本机路径（...\AIDSi\AIDO\vertical_slice\data\） | → GPU 机路径 |
|---|---|
| `adamson/perturb_processed.h5ad`（579M） | `~/map_rna/data/adamson/` |
| `norman/perturb_processed.h5ad`（2.1G） | `~/map_rna/data/norman/` |
| `replogle_rpe1_essential/perturb_processed.h5ad`（2.7G） | `~/map_rna/data/replogle_rpe1_essential/` |

## ④ 传完后在 GPU 机执行（目录校验 + CodeBuddy 开跑）

```bash
# 目录应长这样
ls ~/map_rna/src/ ~/map_rna/weights/ ~/map_rna/data/*/ ~/map_rna/MAP-KG-main/ ~/map_rna/gpu_pack/

# CodeBuddy 开跑指令（贴给它）：
# "读 ~/map_rna/gpu_pack/TASK_P1_001.md 和 CODEBUDDY_BRIEF.md，按任务卡执行冒烟（2 epoch），
#  成功后写 REPORT 进 ~/map_rna/outbox/ 并立即回传，然后继续 30 epoch 全量。"
```

## 回传（GPU → 本机）

| GPU 机路径 | → 本机路径 |
|---|---|
| `~/map_rna/outbox/`（REPORT.md + train_log.jsonl + log 尾 50 行 + debug_log.md + fix.bundle + ckpt_p1_best.pt） | `D:\WorkBuddy\alphafold-web\AIDSi\VCPE_rna\results\inbox_p1\` |
