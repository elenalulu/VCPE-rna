# P1 GPU 作业：MAP-rna 敲低条件化训练（对打 17-result）

## 拷什么 → 哪台机

| 内容 | 本机路径 | GPU 机落位建议 |
|---|---|---|
| 代码 | `VCPE_rna/src/maprna_p1/`（3 个 py） | `~/map_rna/src/` |
| 上游 repo（已打补丁） | `AIDSi/MAP-KG-main/`（整个目录，含 git） | `~/map_rna/MAP-KG-main/` |
| 权重三件 | `VCPE_rna/data/drive_weights/` | `~/map_rna/weights/` |
| 训练数据 | `AIDSi/AIDO/vertical_slice/data/{adamson,norman,replogle_rpe1_essential}/perturb_processed.h5ad` | `~/map_rna/data/` |

## 环境（GPU 机，conda env trRNA2 可复用或新建）

```bash
pip install anndata omegaconf  # 其余 MAP 依赖见 MAP/requirements.txt
```

## 运行

```bash
cd ~/map_rna/MAP-KG-main/MAP   # imports 依赖 cwd/PYTHONPATH 在 MAP/ 下
export MAP_KG_ENCODER_CKPT=~/map_rna/weights/mapkg_encoder_v3.pt
export MAP_ALL_EMBEDDINGS=~/map_rna/weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt

python ~/map_rna/src/train_p1.py \
  --data_dirs ~/map_rna/data/adamson/perturb_processed.h5ad \
              ~/map_rna/data/norman/perturb_processed.h5ad \
              ~/map_rna/data/replogle_rpe1_essential/perturb_processed.h5ad \
  --esm_table $MAP_ALL_EMBEDDINGS \
  --se_ckpt   ~/map_rna/weights/se600m.safetensors \
  --se_config configs/se600m.yaml \
  --map_repo  ~/map_rna/MAP-KG-main/MAP \
  --out_dir   ~/map_rna/p1_out --epochs 30 --amp --batch_size 4 \
  2>&1 | tee ~/map_rna/p1_train.log
```

注意：`se600m.safetensors` 由本机从 epoch_3.pt 抽取（已完成，在 drive_weights/ 里一并拷过去）。

## 看什么

- `p1_out/train_log.jsonl`：逐 epoch train loss + EVAL 行
- 对打基线（frozen 17-result）：`mse_DE 0.5539 · vs ctrl +27.0% · vs mean +11.3% · FM cosine 0.984`
- **G2 门槛**：`mse_DE ≤ 0.56` 且 `fm_cosine ≥ 0.97`
- ⚠️ 注意：P1 的 mse_DE 在 HVG-2000 空间按 perturbation-level holdout 计算；与 17-result 的评估空间/划分
  不完全同源（前者我们的协议、后者 vertical_slice 协议）。对打时两列数字都打印，判读以**相对关系**
  （MAP-rna vs 各 baseline 的差值方向）为主，绝对值差异在 P1_result 阶段做协议对齐后再下结论。
