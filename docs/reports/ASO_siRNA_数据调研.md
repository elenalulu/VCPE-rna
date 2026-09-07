# ASO / siRNA 可用数据调研（2026-09-03，三路检索实证）

> 目的：为 VCPE-rna 的"序列模态扩展"（Tier 2/3）和数据扩容专项摸清 ASO/siRNA 的公开数据版图。
> 结论先行：**ASO 侧有一个重磅发现（ASO Atlas，18.8 万条带化学修饰的效力数据）；siRNA 侧
> 效率预测数据成熟（经典基准齐备）；两者与转录组响应数据的连接点是我们已有的链式架构。**

---

## 一、ASO 数据（三类，用途不同）

### A1. 序列/化学修饰 → 效力（Tier 2/3 的入场券）⭐⭐⭐

| 资源 | 规模 | 内容 | 出处/获取 |
|---|---|---|---|
| **ASO Atlas** ⭐ | **188,521 条 gapmer ASO**，334 个靶基因，417 份 USPTO 专利（2001-2025） | ASO 序列 + **化学修饰标注**（52.6% 5-10-5 2'-MOE gapmer、30.4% 3-10-3 cEt、PS 骨架模式）+ 靶基因 + qRT-PCR 敲低效力；多细胞系（A-431 44,855 / HepG2 23,428 / SH-SY5Y 18,192）；效力谱全（中位 45%，17.4% <10% 的无效序列也在） | OligoAI 论文（bioRxiv 2025-10.29.685292），配套 ML 方法已发表 |
| **ASOptimizer 数据库** | **187,090 条**，67 个靶 mRNA | 文献+专利全量：序列、修饰类型/位置、**剂量（nM）、处理时长（h）、细胞系、转染条件**、qRT-PCR 抑制率 | Molecular Therapy NA 2024（ASOptimizer 论文），平台开源 |
| **RNATx-Bench ASO 任务** | 34,773 条已验证 ASO | 4 个临床相关靶点（SOD1/K-RAS/APOL1/SNHG14）的序列 + 抑制率回归标签 | RNAGenesis 论文（bioRxiv 2024.12.30）基准数据 |

**用途**：训练"**ASO 序列+修饰 → 靶基因敲低深度**"模块——这是 ASO 设计的直接优化目标
（对应我们商业分析里"siRNA 序列效率为 priority"的 ASO 版）。**注意**：这是单靶点 qPCR 效力，
不是转录组响应——它喂的是"效率调制器"，与 VCPE 的响应预测互补而非替代。

**合规提醒**：ASO Atlas / ASOptimizer 数据源自**专利文书**（USPTO/Lens 公开）。专利文本本身
公开，但**用于训练的数据再分发与商用边界需做一次法务复核**——这是本专项唯一需要合规审查的点。

### A2. ASO 转录组响应（Tier 1 验证集 / 日后微调）⭐⭐

| GEO 编号 | 内容 | 价值 |
|---|---|---|
| **GSE293987**（Ionis，2025-11 公开） | **CR-DGE 浓度梯度工作流**：A431 细胞、ACTN1 gapmer ASO + 非靶向对照、**浓度-响应** 3'Tag-Seq，53 样本；Nucleic Acid Ther 2025 | **剂量维度 + Ionis 官方方法学**——ASO 转录组响应的黄金标准协议；验证我们响应预测的浓度依赖性 |
| **GSE289964**（2025-03） | **体内**：小鼠肝，3 种修饰 gapmer ASO（Scarb1）+ PBS，**day 1/3/7 时间 course**，48 样本 | 罕见的 in vivo + 多修饰 + 时间维度 |
| **PRJNA787253**（2021） | Huh-7 细胞，LNA-gapmer **脱靶评估**，微阵列，n=4/组；结论：脱靶程度与互补度相关 | **脱靶建模直接数据**——ASO 序列相似度 → 脱靶基因集，接我们的 off-target 管线 |
| **GSE183535**（2021） | HeLa，MYC gapmer ASO，10nM，4h/18h，21 样本 | 小而干净的时间 course |

**规模判断**：转录组级 ASO 数据总量小（个位数 GSE、几百样本）——只够做**验证集**（检验
CRISPR→ASO 迁移偏差）和日后小规模微调，撑不起主训练。

### A3. ASO 脱靶序列规律
- PRJNA787253 的"互补度→脱靶"定量结论 + CR-DGE 的选择性谱系 → 直接增强我们
  `aso_offtarget_transcriptome.py` 的打分规则（seed 互补度加权）。

---

## 二、siRNA 数据

### B1. 效率预测（序列→沉默效率）——成熟基准齐备 ⭐⭐

| 数据集 | 规模 | 说明 |
|---|---|---|
| **Huesken 2005** | 2,361 条 | 经典基准（24mer，小鼠/人 34 基因），几乎所有 siRNA 效率模型的训练集 |
| **Takayuki** | 702 条 | RNAGenesis 基准第二集 |
| **shRNA 效率集** | 2,076 条 | RNAGenesis shRNA 任务（连续标签） |
| siRNA 设计规则文献 | — | 2004-2012 的规则时代 + 后续 ML（含 off-target 变量），沉淀充分 |

**用途**：siRNA 效率模块（对应记忆里"siRNA 序列效率为 priority"）——这两个基准加上
ASO Atlas 的方法学，可以直接开工。RNAGenesis 已把评测协议（AUROC/AUPRC）定好，对标方便。

### B2. RNAi 转录组响应（Tier 2 微调的起步数据）⭐⭐

| 资源 | 规模 | 关键点 |
|---|---|---|
| **LINCS GSE106127**（CMap RNAi 评估） | **13,000+ shRNA / 9 细胞系**（L1000）+ 373 sgRNA / 6 细胞系 | Levels 4/5 齐全；**CGS（共识基因签名，Level 6）**：同基因多 shRNA 合并 ≈ 4,370 条可用 profile |
| **GSE92742 / GSE70138**（LINCS Phase 1/2） | 5,806 遗传扰动（shRNA KD + OE） | 978 landmark 基因 + 可推演 ~10k BING 基因 |

**⚠️ 最重要的诚实警示（来自 CMap 官方分析）**：shRNA 的 **miRNA-like seed 脱靶效应"比
通常认知的更强且更普遍"**——共享 seed 的 shRNA 对相似度**超过**同基因不同 shRNA 的相似度。
这对 siRNA 同样成立（同一机制）。含义：
1. shRNA/siRNA 数据里"响应"混着大量 seed-driven 脱靶信号——直接当 on-target 响应训练集有系统偏差
2. 好消息：我们的 CRISPR 数据（X-Atlas/PerturbDB）**没有** seed 问题——CRISPR 转录组响应
   更干净，这反过来加强了"CRISPR 学响应 + ASO Atlas 学效力"的分工
3. LINCS CGS 用作 Tier 2 时，建议先用我们已有的 off-target 管线过滤 seed 匹配

---

## 三、拼图：ASO/siRNA 数据怎么接进 VCPE 架构

```
                    ┌─ CRISPR Perturb-seq（X-Atlas/PerturbDB，扩容中）
                    │   → 响应半边：靶基因敲低 → 全转录组响应（已验证 pearson_dev 0.31）
                    │
VCPE-rna 引擎 ──────┼─ ASO Atlas 188k / RNATx-Bench 34.8k
                    │   → 效力半边：ASO 序列+修饰 → 靶基因敲低深度（剂量/时间可调）
                    │        ⚠️ 专利来源，商用前法务复核
                    │
                    └─ GEO ASO 转录组 GSE ×4
                        → 验证集：检验 CRISPR→ASO 机制迁移偏差（RNase H1 vs DNA 层面）
                        → off-target 规则强化（互补度→脱靶）
```

**预测形态（Tier 2 全链）**：用户给 ASO 序列 → off-target 管线给靶谱 → 响应引擎给逐靶转录组
响应 → ASO Atlas 训练的效力头给**敲低深度调制**（含剂量）→ 合成最终预测。每一块的数据都
已找到公开来源。

---

## 四、行动清单（建议优先级）

| # | 动作 | 成本 | 产出 |
|---|---|---|---|
| 1 | 下载 ASO Atlas（OligoAI 配套）+ RNATx-Bench，摸清格式 | 半天 | 效力模块的数据就位 |
| 2 | ASO 效力头 v1（序列+修饰 → 抑制率，对照 RNAGenesis 协议） | 1-2 天 | siRNA/ASO 序列效率 priority 落地 |
| 3 | 摄入 4 个 ASO 转录组 GSE → ASO 验证集构建 | 1 天 | CRISPR→ASO 迁移偏差首次实测 |
| 4 | 法务复核：专利衍生数据（ASO Atlas/ASOptimizer）商用边界 | 外部/自查 | 合规结论 |
| 5 | LINCS CGS 4,370 评估（seed 过滤后）作 Tier 2 微调池 | 1 天 | 决定 Tier 2 主训练集构成 |

---

## 附：检索命令与来源

- 检索时间：2026-09-03，三路（ASO 转录组 / siRNA LINCS / 效率与修饰数据库）
- 关键来源：bioRxiv 2025.10.29.685292（OligoAI/ASO Atlas）、Mol Ther NA 2024（ASOptimizer）、
  bioRxiv 2024.12.30（RNAGenesis/RNATx-Bench）、NCBI GEO（GSE293987/GSE289964/GSE183535/
  GSE106127/GSE92742/GSE70138）、PRJNA787253
