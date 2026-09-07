# CODEBUDDY 交接简报：MAP-rna P1 GPU 训练作业

> 你（CodeBuddy）在 Linux GPU 机器上负责：跑通、调试、产出训练指标。
> 你**不负责**：修改评估协议、修改模型架构设计、修改 G2 门槛——遇到这三类问题停下来，把情况写进 `debug_log.md` 交给项目负责人（通过用户传回）。

## 1. 任务一句话

训练 `train_p1.py`（敲低条件化的 MAP 扰动响应模型），在联合 Norman/Adamson/Replogle Perturb-seq 上收敛，
产出逐 epoch 指标，供与既有基线对打。目标是**跑通并收敛**，不是调到最优。

## 2. 环境事实（已验证的部分）

- GPU：单卡 4090 Ada 24G；conda env `trRNA2`（Py3.10, openmm CUDA）可复用，或新建 env
- Python 依赖（除下述外见 `MAP-KG-main/requirements.txt`）：`anndata`、`omegaconf` 必装
- 权重（应已在 `~/map_rna/weights/`）：
  - `epoch_3.pt`（4.23GB，上游全量 ckpt）
  - `se600m.safetensors`（已从 epoch_3.pt 抽取的 SE 底座权重，234 keys）
  - `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`（411MB，基因符号→5120 维嵌入表，19790 基因）
  - `mapkg_encoder_v3.pt`（1.13GB，KG 编码器）
- 数据（应已在 `~/map_rna/data/`）：三份 GEARS 格式 `perturb_processed.h5ad`
  （adamson / norman / replogle_rpe1_essential）
- 上游 repo `MAP-KG-main/` **已带 10 个补丁（git 管理，4 commits）**——不要 git reset，不要覆盖以下文件：
  `MAP/model/model.py`、`MAP/model/pert.py`、`MAP/model/transformer_encoder.py`、
  `MAP/model/mega_molbart/STMencoder_ddp.py`、`MAP/demo.py`。补丁内容见 `git log` 三条 fix commit。

## 3. 运行命令

```bash
cd ~/map_rna/MAP-KG-main/MAP
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

**先跑冒烟**（1-2 个 epoch 验证全链路，再放 30）：
`--epochs 2 --batch_size 2 --num_workers 2`

## 4. 已知坑（本机已踩过，GPU 侧大概率还会遇到）

| 坑 | 症状 | 处置 |
|---|---|---|
| 显存 OOM | CUDA OOM | 先降 `--batch_size`（最小 1），再降 `--set_size`（最小 4）；SE 已冻结，不要解冻省显存的反向操作 |
| bf16 NaN | loss 变 NaN | 去掉 `--amp` 跑 fp32（上游 eval 阶段同样有此问题，属已知） |
| dataloader 卡死 | num_workers>0 卡住 | `--num_workers 0` 排查 |
| h5ad 内存 | 加载 3 份 h5ad RAM 高 | 正常现象，机器 RAM≥64G 即可；仍爆则逐份加载调 `--data_dirs` |
| 靶基因不在 ESM2 表 | `pert_row=-1` 的样本 | 数据集已自动跳过（sym2row miss 会打印计数），若 miss 率 >10% 记入 debug_log 并继续，不要自己加映射表 |
| `se600m.safetensors` 不存在 | FileNotFoundError | 从 `epoch_3.pt` 抽取：见 `~/map_rna/src/` 同目录说明，或回传报错 |

## 5. 你的 debug 循环协议

1. **跑** → 读 traceback → **根因定位**（读代码，别盲试）→ 最小修复 → 重跑
2. 每次修复：`git add -A && git commit -m "fix(P1-gpu): <一句话>"`（repo 已 git init）
3. 每次修复同步记一行到 `~/map_rna/debug_log.md`：`#N | 症状 | 根因 | 修法 | 文件`
4. 允许自己修：环境/依赖/路径/OOM/显存/数据加载/明显笔误
5. **必须上报、不许自己决定**：
   - 评估指标计算逻辑（train_p1.py 的 evaluate 函数）
   - 模型结构（kd_projector / 条件化方式 / 冻结策略）
   - 数据划分（holdout 逻辑）
   - G2 门槛数字
   - 任何"改了能跑但不确定对不对"的动静——宁可停下来问
6. 连续 3 次修复同一问题未解决 → 停止，整理现场等指示

## 6. 完成标准与回传物

跑满 `--epochs`（或 2 epoch 冒烟成功即先回传一次）。回传四样：

1. `p1_out/train_log.jsonl` 全文（训练 loss + 每行 EVAL 指标）
2. `p1_train.log` 最后 50 行
3. `git log --oneline`（GPU 侧新 commit 列表）+ `git diff <起点>..HEAD --stat`
4. `debug_log.md`

**指标判读由项目负责人做**（对照基线 mse_DE 0.5539 / FM cosine 0.984 / G2 门槛 mse_DE≤0.56 且 cosine≥0.97），
你只负责把数字如实带回来，不做好坏结论。

## 7. 背景速览（30 秒版）

MAP-rna = 把 MAP（知识图谱对齐+条件化扰动预测，MIT 许可）的药物条件化换成"靶基因敲低"条件化，
用于 ASO/siRNA 敲低的全转录组响应预测，目标替代现有 GenBio Non-Commercial 底座的管线。
P1 = 验证 MAP 底座在我们的 Perturb-seq 数据上能达到现有生产模型的效果水平（G2）。
模型细节：SE 细胞编码器冻结（600M），训练的是扰动通路 + 新增 kd_projector（ESM2 5120→1024）。

## 8. 协作协议（离线优先，2026-08-30 增补）

- **任务来源**：每轮作业以 `gpu_pack/TASK_*.md` 任务卡为准（卡里引用本总纲）。没有任务卡的新工作不要自行开始。
- **回传协议**：所有产出写进 `~/map_rna/outbox/`（REPORT.md + 日志 + git bundle），用户打包带回。
- **离线 git 同步**：GPU 机连不上 GitHub 也没关系——每次修复后
  `git bundle create <outbox>/fix.bundle main`，bundle 由用户带回本机合并（历史无损）。
  GitHub push 仅在方便时做（可选镜像，非必需）。
- **inbox/outbox 约定**：用户放进 `~/map_rna/inbox/` 的 zip = 新任务卡+代码更新，解包后按卡执行。
