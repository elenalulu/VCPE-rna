# VCPE-rna

**V**irtual-**C**ell **P**erturbation-response **E**ngine — RNA branch.
一个许可干净（全链 MIT/Apache-2.0）的**敲低扰动响应预测引擎**：给定敲低扰动（CRISPRi/siRNA/ASO 靶基因），预测转录组层面的响应方向与幅度。定位为虚拟细胞技术栈中的**扰动响应引擎层**（伪 bulk / 转录组模态），可作为 Non-Commercial 权重（如 AIDO.RNA-Pert）的可商用替代。

## 核心结果

| 阶段 | 关键数字 | 状态 |
|---|---|---|
| P1（MAP 底座 + 敲低条件化） | mse_DE 0.0215（ctrl 基线好 32%）· fm_cosine 0.9604 | ✅ G2 条件通过 |
| P2（双轴：ESM2+KG ⊕ 自训 RNA 编码器） | fm_cosine 0.9710 —— **后被勘误撤回**：条件化被共性响应 shortcut 架空 | ⚠️ 见勘误 |
| P2.3-B（GEARS 式逐基因残差头） | **条件化复活**：ablation_r 1.000 → 0.277，pearson_dev 0.3079，训练仅 80 秒 | ✅ G2''' 达标 |
| 数据扩容（+scPerturb 6 数据集） | 扰动数 1,843 → 16,276（×8.8），LEGACY 域 pearson_dev **0.31 → 0.71**，overall 0.37 | ✅ 通过 |
| ASO 效力头 v3（ASO Atlas，patent-split） | pooled Spearman 0.283 / per-screen median 0.261（含 GENCODE region 特征 +0.034） | 与 OligoAI 基线可比 |
| siRNA 效力头 v1（Huesken 2,431 条） | CV Spearman 0.607 · 外部 Ichihara_2007_2 Spearman 0.588 | 对标 RNAGenesis 基准 |

> **科学诚实声明**：P2 的 fm_cosine 0.9710 一度被判为达标，后经三路诊断（`diag_fc.py`）证明模型学到的是数据集级**共性响应**（K562 敲低共享转录组签名），条件化被训练目标中的共性核架空（ablation_r = 1.000）。全部四项常规指标（mse/pearson/cosine/top50）都可被共享成分灌水——这是本次开源最重要的方法论教训之一，详见 [docs/reports/P2_result.md](docs/reports/P2_result.md) 勘误章节与 [docs/reports/P2.3-B_result.md](docs/reports/P2.3-B_result.md)。

## 全链架构

```
                    ┌─────────────────────────────────────────────┐
                    │  条件化输入                                  │
                    │  机制轴: 靶基因 ESM2 嵌入 ⊕ MAP-KG 对齐      │
                    │  序列轴: 自训 RNA 编码器 (2.5M, InfoNCE)     │
                    │  网络轴: STRING 邻居指示 + is_target         │
                    └──────────────────┬──────────────────────────┘
                                       ▼
   Perturb-seq h5ad ──► 逐基因偏差头 DeviationModel ──► pred_dev (residual)
   (adamson/norman/replogle    (P2.3-B, 5.7M params)        │
    + scPerturb ×6)                                          ▼
                                              pred_fc = pred_dev + common_fc
                                                       (train-only 共性核)
                                       ▼
                    vcpe_cache 导出（批量推理 → AIDO 缓存同构 schema）
                                       ▼
                    平台零依赖接入（换缓存文件即完成替换）
```

横向模块（与主链并行）：

- **L2 RNA 编码器扩容**（`src/l2/`）：RNAcentral 45M → 20-500nt 过滤去重 → MLM 预训练 → 对比对齐到 ESM2 嵌入空间
- **效力头**（`src/efficacy/`）：ASO 序列+化学修饰 → 敲低深度（ASO Atlas 19 万条）；siRNA guide → 效率（Huesken 基准）；供 reagent design 与响应先验使用
- **数据扩容**（`src/data_expansion/` + `src/maprna_p3/ingest_*.py`）：scPerturb Zenodo 摄入 → 伪 bulk 处理

## 全链证据

| 报告 | 内容 |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | 项目规划 v0.3：D1-D3 决策、阶段计划与 go/no-go、基准协议、许可清单 |
| [docs/reports/P1_result.md](docs/reports/P1_result.md) | P1 G2 判定：13 epoch 轨迹、AMP 尖峰诊断 |
| [docs/reports/P2_result.md](docs/reports/P2_result.md) | P2 终判 + **勘误**：共性响应 shortcut 的完整证据链 |
| [docs/reports/P2.3-B_result.md](docs/reports/P2.3-B_result.md) | 条件化复活：架构为什么这次成了（逐基因直出 vs 加性 token） |
| [docs/reports/数据扩容专项验收报告.md](docs/reports/数据扩容专项验收报告.md) | ×8.8 扩容后分域对打表与域梯度解读 |
| [docs/reports/P3_接入分析.md](docs/reports/P3_接入分析.md) | 平台接入设计：离线缓存模式，零新依赖 |
| [docs/reports/ASO_siRNA_数据调研.md](docs/reports/ASO_siRNA_数据调研.md) | ASO/siRNA 公开数据版图（ASO Atlas 等） |
| [docs/reports/L2训练.md](docs/reports/L2训练.md) | L2 编码器预训练记录 |

## 仓库结构

```
src/
  maprna_p1/   P1 训练（MAP 底座 + 敲低条件化；需 patched MAP repo + SE 权重）
  maprna_p2/   P2 双轴训练 + RNA 序列编码器（InfoNCE 对齐）+ 三路诊断 diag_fc
  maprna_p3/   P2.3-B 逐基因偏差头（独立训练，无 SE/MAP 依赖）+ 数据摄入 + 公平评测
  l2/          RNAcentral 下载 → 过滤去重 → tokenize → MLM 预训练 → 对比对齐
  efficacy/    ASO/siRNA 效力头（ASO Atlas / OligoGym / Huesken）+ 预测服务
  data_expansion/  scPerturb Zenodo 下载与摄入
  export_vcpe_cache.py   平台缓存导出（AIDO 缓存同构 schema）
docs/          PLAN、全链报告、GPU 任务卡（gpu_pack/）
results/       训练日志、CV 结果、评测产物（大 ckpt 不入库，见下）
data/          数据获取指南（README.md，数据本体不入库）
```

## Quickstart

P3 响应头**不依赖** SE 底座 / MAP repo / flash-attention，单卡或 CPU 数分钟可复现：

```bash
pip install -r requirements.txt

# 1. 准备数据（见 data/README.md）：
#    - Perturb-seq h5ad（GEARS 格式：adamson / norman / replogle_rpe1_essential）
#    - ESM2 基因嵌入表 Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt

# 2. 训练 P3 逐基因偏差头（60 epochs ≈ 80 秒 @GPU；CPU 亦可）
python src/maprna_p3/train_p3.py \
  --data_dirs data/adamson/perturb_processed.h5ad \
              data/norman/perturb_processed.h5ad \
              data/replogle_rpe1_essential/perturb_processed.h5ad \
  --esm_table data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --out_dir ./p3_out --epochs 60

# 3. 完整链路（P1/P2 需要 MAP 底座 + SE 权重，单卡 24G）
#    命令模板见各脚本 docstring 与 docs/PLAN.md
```

**预训练产物**：`results/p3_v21e/` 内含对齐后的 RNA 序列编码器（`rna_enc_from_ckpt.pt`，9MB）与全基因向量缓存示例。完整 ckpt（P3 响应头 428MB、效力头 ~410MB 等）计划发布于 GitHub Releases / HuggingFace，不入 git。

## 许可与归属

- 本仓库代码：**MIT License**
- 架构上游：[MAP](https://github.com/MAGIC-AI4Med/MAP)（MIT）+ MAP-KG（Apache-2.0）；P2.3-B 偏差头为独立自研
- 数据许可与获取方式逐项见 [data/README.md](data/README.md)
- ⚠️ **ASO Atlas 为 USPTO 专利衍生数据**，本仓库只含训练/评测代码不含数据；任何商用前需独立法务复核
- AIDO.RNA-650M（Non-Commercial）仅限内部研究消融，不进任何交付物——这也是本项目存在的理由

## 方法论教训（选摘）

1. **评估必须含"去共享成分后的判别力"指标**（pearson_dev / top50_dev + zeroPert ablation），否则 G-gate 会被共性响应骗过（P2 教训）
2. **加性 token 进大主干残差流会被结构性淹没**——数据受限场景下逐基因直出残差头是正确架构（P2.3-B 结论）
3. **数据扩容后 LR 需重标定**：×8.8 数据 + lr 1e-3 会塌缩进零残差盆地，lr 3e-4 全程零失稳
