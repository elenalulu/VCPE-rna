"""P1 training: MAP-rna knockdown-conditioned perturbation predictor.

Single-GPU (4090 24G). Freeze the SE cell encoder (600M), train the
perturbation pathway + KD projector on joint Norman/Adamson/Replogle.
Eval per epoch on held-out perturbations: mse_DE / pearson-delta /
top-50 DEG overlap / FM-cosine, plus ctrl & perturb-mean baselines.

Run (GPU box):
  python train_p1.py \
    --data_dirs <AIDO>/vertical_slice/data/adamson/perturb_processed.h5ad \
                <AIDO>/vertical_slice/data/norman/perturb_processed.h5ad \
                <AIDO>/vertical_slice/data/replogle_rpe1_essential/perturb_processed.h5ad \
    --esm_table <drive_weights>/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --se_ckpt   <drive_weights>/se600m.safetensors \
    --se_config <MAP repo>/configs/se600m.yaml \
    --map_repo  <MAP repo>/MAP \
    --out_dir   ./p1_out --epochs 30 --amp
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# NOTE: model_kd / ds_knockdown are imported INSIDE main(), after
# sys.path.insert(0, args.map_repo) — they need the patched MAP repo on the
# path before `from model.model import ...` resolves.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", type=str, nargs="+", required=True,
                   help="paths to perturb_processed.h5ad files")
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--se_ckpt", type=str, required=True)
    p.add_argument("--se_config", type=str, required=True)
    p.add_argument("--map_repo", type=str, required=True,
                   help="path to the (patched) MAP repo MAP/ directory")
    p.add_argument("--out_dir", type=str, default="./p1_out")
    p.add_argument("--set_size", type=int, default=8)
    p.add_argument("--sent_len", type=int, default=2048)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_kd", type=float, default=5e-4, help="lr for the new KD projector")
    p.add_argument("--num_warmup_steps", type=int, default=200)
    p.add_argument("--hvg_loss_weight", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval_every_epochs", type=int, default=1)
    return p.parse_args()


def split_perturbations(items, test_frac, seed):
    """Deterministic perturbation-level holdout, stratified per dataset."""
    rng = np.random.default_rng(seed)
    by_ds = {}
    for it in items:
        by_ds.setdefault(it[0], []).append(it)
    train, test = [], []
    for di, lst in sorted(by_ds.items()):
        lst = sorted(lst)
        idx = rng.permutation(len(lst))
        n_test = max(1, int(round(len(lst) * test_frac)))
        test.extend([lst[i] for i in idx[:n_test]])
        train.extend([lst[i] for i in idx[n_test:]])
    return train, test


def precompute_pert_emb_targets(model, kd, items, device, batch=32):
    """SE embedding of each perturbation's perturbed bulk (frozen SE, one pass).

    Returns dict (ds, cond) -> np.ndarray [2048].
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            for di, cond in chunk:
                idx = None
                # build perturbed bulk sentence the same way the ctrl sentence is built
                rows = kd["row_of_gene"][di]
                Xp = kd["X_pert"][di]
                cp = kd["pert_labels"][di]
                sel = np.where(cp == cond)[0]
                Xb = Xp[sel].mean(axis=0)                       # [n_genes] mean profile
                m = rows >= 0
                gid, ex = rows[m], Xb[m]
                order = np.argsort(-ex)[: 2047]
                # [1, L] (B=1, single sentence) — pe_embedding gives [1, L, 5120], 3-dim,
                # matching the cls_token's 3-dim expand below (bug: was [None, None, :], 4-dim)
                ids = np.concatenate([[0], gid[order]]).astype(np.int32)[None, :]
                exs = np.concatenate([[0.0], ex[order]]).astype(np.float32)[None, :]
                src = model.se.pe_embedding(torch.from_numpy(ids).long().to(device))
                src = torch.nn.functional.normalize(src, dim=2)
                cls = model.se.cls_token.expand(src.size(0), 1, -1)
                src = torch.cat([cls, src[:, 1:, :]], dim=1)
                if model.se.dataset_token is not None:
                    src = torch.cat((src, model.se.dataset_token.expand(src.size(0), 1, -1)), dim=1)
                with torch.no_grad():
                    _, emb, _ = model.se(src=src,
                                         counts=torch.from_numpy(exs).float().to(device),
                                         dataset_nums=None, profile=False)
                out[(di, cond)] = emb.reshape(emb.shape[0], -1)[0].float().cpu().numpy()
    return out


@torch.no_grad()
def evaluate(model, loader, device, hvg_rows, kd, ctrl_means, pert_mean_global):
    """Metrics on held-out perturbations (HVG space + embedding cosine)."""
    model.eval()
    mse_list, pear_list, top50_list = [], [], []
    cos_list = []
    base_ctrl_mse, base_mean_mse = [], []
    for batch in loader:
        ctrl_gene = batch["control_gene_ids"].to(device)
        ctrl_expr = batch["control_expressions"].to(device)
        pert_row = batch["pert_row"].to(device)
        true_hvg = batch["pert_hvg"].numpy()               # [B, 2000]
        with torch.autocast("cuda", enabled=False):
            pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, pert_row)
        pred_hvg = pred_hvgs.float().mean(dim=1).cpu().numpy()   # [B, 2000]

        ctrl_profiles = [kd["X_ctrl"][di].mean(axis=0) for di in batch["ds"]]
        ctrl_hvg = np.stack([
            np.array([ctrl_profiles[b][np.where(kd["row_of_gene"][batch["ds"][b]] == int(hr))[0][0]]
                      if (kd["row_of_gene"][batch["ds"][b]] == int(hr)).any() else 0.0
                      for hr in hvg_rows]) for b in range(len(batch["ds"]))])
        pert_mean_hvg = np.array([
            [pert_mean_global.get((int(hr)), 0.0) for hr in hvg_rows]
            for _ in range(len(batch["ds"]))])

        true_fc = true_hvg - ctrl_hvg
        pred_fc = pred_hvg - ctrl_hvg
        base_fc_ctrl = np.zeros_like(true_fc)
        base_fc_mean = pert_mean_hvg - ctrl_hvg

        mse_list.extend(np.mean((pred_fc - true_fc) ** 2, axis=1))
        base_ctrl_mse.extend(np.mean((base_fc_ctrl - true_fc) ** 2, axis=1))
        base_mean_mse.extend(np.mean((base_fc_mean - true_fc) ** 2, axis=1))

        for b in range(len(true_fc)):
            t, p = true_fc[b], pred_fc[b]
            if t.std() > 1e-6 and p.std() > 1e-6:
                r = float(np.corrcoef(t, p)[0, 1])
            else:
                r = 0.0
            pear_list.append(r)
            k = min(50, len(t))
            t50, p50 = set(np.argsort(-np.abs(t))[:k]), set(np.argsort(-np.abs(p))[:k])
            top50_list.append(len(t50 & p50) / k)

        # FM cosine: predicted embedding vs SE embedding of true perturbed bulk
        tgt = batch.get("pert_emb")
        if tgt is not None:
            pe = pred_embs.float().mean(dim=1).cpu().numpy()
            tg = tgt.numpy()
            cos_list.extend(np.sum(pe * tg, axis=1) /
                            (np.linalg.norm(pe, axis=1) * np.linalg.norm(tg, axis=1) + 1e-8))

    res = dict(
        mse_DE=float(np.mean(mse_list)),
        pearson_delta=float(np.mean(pear_list)),
        top50_deg_overlap=float(np.mean(top50_list)),
        mse_DE_baseline_ctrl=float(np.mean(base_ctrl_mse)),
        mse_DE_baseline_perturbmean=float(np.mean(base_mean_mse)),
        n_eval=len(mse_list),
    )
    if cos_list:
        res["fm_cosine"] = float(np.mean(cos_list))
    return res


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)

    sys.path.insert(0, args.map_repo)
    from omegaconf import OmegaConf  # noqa: E402
    from model_kd import MAPmodelKD, trainable_parameter_report  # noqa: E402
    from ds_knockdown import (build_symbol2row, load_kd_datasets,  # noqa: E402
                              make_hvg_list, KnockdownDataset, collate_kd)

    # data
    sym2row, esm_dim = build_symbol2row(args.esm_table)
    print(f"ESM2 table: {len(sym2row)} symbols, dim {esm_dim}", flush=True)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)
    print(f"HVG rows: {len(hvg_rows)}", flush=True)

    # perturbation-level split
    train_items, test_items = split_perturbations(kd["pert_index"], args.test_frac, args.seed)
    print(f"perts: train={len(train_items)} test={len(test_items)}", flush=True)

    ds_train = KnockdownDataset(kd, train_items, sym2row, set_size=args.set_size,
                                L=args.sent_len, hvg_rows=hvg_rows, is_train=True, seed=args.seed)
    ds_test = KnockdownDataset(kd, test_items, sym2row, set_size=args.set_size,
                               L=args.sent_len, hvg_rows=hvg_rows, is_train=False, seed=args.seed)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_kd, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_kd)

    # model
    cfg = OmegaConf.load(args.se_config)
    model = MAPmodelKD(se_ckpt=args.se_ckpt, se_cfg=cfg,
                       esm_table_path=args.esm_table, hvg_info=None, freeze_se=True)
    model = model.to(device)

    # SE frozen; also freeze the unused KG-SMILES encoder; train the rest
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("pert_model.") and "kg_smiles_encoder" not in name:
            p.requires_grad = True
    total, trainable = trainable_parameter_report(model)
    print(f"params total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M", flush=True)

    kd_params = [p for n, p in model.named_parameters() if "kd_projector" in n and p.requires_grad]
    base_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "kd_projector" not in n]
    optimizer = torch.optim.AdamW([
        dict(params=base_params, lr=args.lr),
        dict(params=kd_params, lr=args.lr_kd),
    ], weight_decay=0.01)

    def lr_lambda(step):
        if step < args.num_warmup_steps:
            return step / max(1, args.num_warmup_steps)
        prog = (step - args.num_warmup_steps) / max(1, 10 ** 9)
        return 1.0  # constant after warmup; cosine optional for short runs
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_emb = nn.MSELoss()
    loss_hvg = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # perturb-mean global baseline (HVG space)
    pert_mean_global = {}
    for di, cond in train_items:
        Xp = kd["X_pert"][di]
        cp = kd["pert_labels"][di]
        m = Xp[np.where(cp == cond)[0]].mean(axis=0)
        rows = kd["row_of_gene"][di]
        for hr in hvg_rows:
            w = np.where(rows == int(hr))[0]
            if len(w):
                pert_mean_global[int(hr)] = float(m[w[0]])

    # precompute SE embeddings of true perturbed bulks (targets, frozen SE)
    print("precomputing SE target embeddings...", flush=True)
    emb_targets = precompute_pert_emb_targets(model, kd, kd["pert_index"], device)
    ds_train.emb_targets = emb_targets
    ds_test.emb_targets = emb_targets

    def wrap_loader(loader, dataset):
        for batch in loader:
            batch["pert_emb"] = torch.stack([
                torch.from_numpy(dataset.emb_targets[(batch["ds"][i], batch["condition"][i])])
                for i in range(len(batch["ds"]))])
            yield batch

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_mse = None
    best_cos = None
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for batch in train_loader:
            ctrl_gene = batch["control_gene_ids"].to(device, non_blocking=True)
            ctrl_expr = batch["control_expressions"].to(device, non_blocking=True)
            pert_row = batch["pert_row"].to(device)
            true_hvg = batch["pert_hvg"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, pert_row)
                emb_loss = loss_emb(pred_embs.float().mean(dim=1),
                                    torch.stack([torch.from_numpy(
                                        emb_targets[(batch["ds"][i], batch["condition"][i])])
                                        for i in range(len(batch["ds"]))]).to(device))
                hvg_loss = loss_hvg(pred_hvgs.float().mean(dim=1), true_hvg)
                loss = emb_loss + args.hvg_loss_weight * hvg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1
            run.append(float(loss.item()))

        msg = dict(epoch=epoch + 1, step=global_step,
                   train_loss=float(np.mean(run)) if run else None,
                   secs=round(time.time() - t0, 1))
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        if (epoch + 1) % args.eval_every_epochs == 0 or epoch == args.epochs - 1:
            res = evaluate(model, wrap_loader(test_loader, ds_test), device,
                           hvg_rows, kd, None, pert_mean_global)
            res.update(epoch=epoch + 1, step=global_step)
            print("[EVAL]", json.dumps(res), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"eval": res}) + "\n")
            # dual-track best saving: G2's headline metric is fm_cosine; mse_DE is
            # the secondary. Save each metric's best separately so a cosine peak
            # is never lost to a slightly-worse mse (seen at epoch 7).
            ck = {"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                  "metrics": res, "hvg_rows": hvg_rows.tolist()}
            if best_mse is None or res["mse_DE"] < best_mse:
                best_mse = res["mse_DE"]
                torch.save(ck, os.path.join(args.out_dir, "ckpt_best_mse.pt"))
                print(f"💾 saved best-mse (mse_DE={best_mse:.4f})", flush=True)
            if "fm_cosine" in res and (best_cos is None or res["fm_cosine"] > best_cos):
                best_cos = res["fm_cosine"]
                torch.save(ck, os.path.join(args.out_dir, "ckpt_best_cosine.pt"))
                print(f"💾 saved best-cosine (fm_cosine={best_cos:.4f})", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "hvg_rows": hvg_rows.tolist()},
               os.path.join(args.out_dir, "ckpt_p1_last.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
