# L2 Pretraining Launch Command (run log)

Pretraining is launched on the GPU machine in the background:

```bash
nohup python $BASE/src/l2/pretrain_l2.py \
  --store_dir $BASE/data/l2 --out_dir $BASE/l2_pretrain \
  --epochs 3 --max_hours 40 \
  >> $BASE/l2_pretrain.log 2>&1 &
```

Inputs come from the token store produced by `tokenize_store.py`
(`l2_tokens.u16` + `l2_offsets.npy` + `l2_train_idx.npy` + `l2_val_idx.npy`).
See [PLAN.md](../PLAN.md) §"L2 Encoder Expansion Kickoff" for the gate definitions
(L2-a pretraining health → L2-b alignment beats L1 → L2-c fm_cosine ≥ 0.975).
