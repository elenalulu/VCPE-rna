# TASK P1-001：冒烟 + 正式训练（MAP-rna 敲低条件化）

> 收到本卡 = 你已读完 `CODEBUDDY_BRIEF.md`（环境/坑/边界/回传规则都在里面，那是总纲，本卡是本轮具体任务）。

## 目标

1. **冒烟**：`train_p1.py` 跑通 2 epoch（`--epochs 2 --batch_size 2 --num_workers 2`）
2. **正式**：冒烟通过后跑 30 epoch 全量（`--epochs 30 --batch_size 4 --amp`）

## 输入清单（缺哪个先报错停，不要自己找替代品）

| 文件 | 建议落位 |
|---|---|
| `src/maprna_p1/{train_p1,ds_knockdown,model_kd}.py` | `~/map_rna/src/` |
| `MAP-KG-main/`（已打补丁的整个上游 repo） | `~/map_rna/MAP-KG-main/` |
| `data/drive_weights/{epoch_3.pt, se600m.safetensors, Homo_sapiens...ESM2.pt, mapkg_encoder_v3.pt}` | `~/map_rna/weights/` |
| `{adamson,norman,replogle_rpe1_essential}/perturb_processed.h5ad` | `~/map_rna/data/` |

## 步骤

1. 环境：GPU 机 `pip install anndata omegaconf`（缺啥装啥，记进 debug_log）
2. 冒烟（命令在 `CODEBUDDY_BRIEF.md` §3，把 `--epochs 2 --batch_size 2` 换进去）
3. 冒烟成功 → **立即写 REPORT 草稿并进 outbox**（不等全量）
4. 全量 30 epoch → 结束写 REPORT 终稿

## outbox 交付物（打包回传）

```
outbox/
├── REPORT.md              # 你的工作汇报：做了什么/修了什么(含 git commit 号)/结果如何
├── train_log.jsonl        # p1_out/ 里的逐 epoch 日志
├── p1_train.log.tail50    # tee 日志最后 50 行
├── debug_log.md           # 每次修复一行（总纲 §5 格式）
├── fix.bundle             # git bundle（见下）
└── ckpt_p1_best.pt        # 最优 ckpt（G2 通过后判读用，体积大可单独传）
```

**离线 git 同步（GPU 机可能连不上 GitHub 也没关系）**：
```bash
cd ~/map_rna/MAP-KG-main && git add -A && git commit -m "..." && git bundle create ../../outbox/fix.bundle main
```
用户把 fix.bundle 带回本机 `git pull fix.bundle main` —— 历史无损合并，全程不需要 GitHub。

## 判读标准（你不用做，只是让你知道为什么跑）

对照基线 mse_DE 0.5539 / FM cosine 0.984；G2 = mse_DE ≤ 0.56 且 cosine ≥ 0.97。数字由项目负责人判读。
