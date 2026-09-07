# 数据获取指南

VCPE-rna **不再分发任何数据本体**（体积与许可原因）。本文件逐项列出各模块所需数据、来源、许可与获取方式。下载脚本见 `src/l2/download_rnacentral.py`、`src/data_expansion/download_scperturb.py`、`src/efficacy/download_oligogym.py`。

## 一、主链：Perturb-seq（GEARS 格式 h5ad）

| 数据集 | 用途 | 来源 | 许可 |
|---|---|---|---|
| adamson / norman / replogle_rpe1_essential `perturb_processed.h5ad` | P1/P2/P3 训练 | [GEARS](https://github.com/snap-stanford/GEARS) 数据发布（各 ~0.5-2.7 GB） | 见 GEARS repo |
| ReplogleWeissman2022 K562 gwps、NadigOConner2024 hepg2/jurkat 等 6 数据集 | 数据扩容专项 | [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497)（`src/data_expansion/download_scperturb.py` 支持 `--all`） | CC BY 4.0 |
| ESM2 基因嵌入表 `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`（392MB） | 条件化查表 | MAP 发布权重（见下） | 按 MAP repo（MIT）推定，商用前开 issue 确认 |
| `gene_transcripts.fa` | RNA 序列轴 | GENCODE v44 transcript FASTA 提取 | GENCODE 许可 |

## 二、MAP 底座权重（仅 P1/P2 需要；P2.3-B 头不需要）

来自 MAP / MAP-KG 发布（[MAGIC-AI4Med/MAP](https://github.com/MAGIC-AI4Med/MAP)，MIT；HF `RainGate/MAP-KG`，Apache-2.0）：

| 文件 | 大小 |
|---|---|
| `epoch_3.pt`（扰动预测器底座） | 4.2 GB |
| `se600m.safetensors`（SE cell encoder） | 2.4 GB |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 392 MB |
| `mapkg_encoder_v3.pt`（KG 编码器） | 1.1 GB |

P1/P2 训练另需 patched MAP repo（本团队打过 10 个工程补丁；补丁要点见 `docs/gpu_pack/`）。

## 三、效力头（src/efficacy/）

| 数据 | 用途 | 来源 | 许可 | ⚠️ |
|---|---|---|---|---|
| ASO Atlas v1/v2（188k gapmer，修饰+效力） | ASO 效力头 | HF `barneyhill/aso-atlas` / `aso-atlas-2`（OligoAI 论文配套） | 见 HF dataset 页 | **USPTO 专利衍生**，商用前须独立法务复核；本仓库不含数据 |
| OligoGym 12 数据集（ichihara/alharbi/huesken 等） | siRNA/ASO 外部评测 | HF `CollageBio/oligo-datasets`（`download_oligogym.py` 直下） | CC BY 4.0 | — |
| GSE183535（MYC ASO RNA-seq）、GSE289964（in vivo 全肝）、GSE293987 | 域迁移评测 | GEO | 见各 series | — |

## 四、L2 RNA 编码器预训练（src/l2/）

| 数据 | 来源 | 许可 |
|---|---|---|
| RNAcentral active fasta（~11 GB，45M seq） | [RNAcentral FTP](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/)（`download_rnacentral.py`，断点续传） | CC0 |
| STRING v12 human edges | [string-db.org](https://string-db.org) | CC BY 4.0 |

## 五、其他

| 数据 | 用途 | 来源 |
|---|---|---|
| GTEx v10 gene median TPM | 原代肝 donor 语境（p05_donor_context） | [gtexportal.org](https://www.gtexportal.org) |
| MAP-KG 知识图谱 CSV | KG 对齐 | HF `RainGate/MAP-KG`（Apache-2.0） |

## 约定

- 默认数据根目录：仓库下 `data/`（git-ignored）；可用 `VCPE_DATA_DIR` 覆盖部分脚本
- 不入库原因：总数据量 ~35 GB + 若干许可不允许再分发
