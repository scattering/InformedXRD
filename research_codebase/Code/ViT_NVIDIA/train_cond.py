"""
Scaling-sweep trainer: conditioned ViT vs unconditioned baseline, same 2-stage curriculum
(uniform pretrain -> rruff-conditioned finetune), evaluated on the REAL RRUFF-473.

Answers the gate question at the 500k+500k scale: does candidate conditioning help EG, and
does the conditioned-minus-baseline delta grow / hold with data scale? Run at several scales
(e.g. 100k/250k/500k per dataset) and read the delta-vs-scale curve.

Bundles (built by assemble_cond.py on Stampede, read from /work2):
  uniform_train.h5, rruff_train.h5  -> X[N,8501], eg[N], peaks, peak_mask, cands, cand_mask
  eval_473.h5 (build_473_eval_bundle.py) -> same fields, real test set

Conditioned vs baseline share everything except: baseline forces conditioning to the null token
(cond_drop=1.0). So the delta is attributable to conditioning, not capacity.
"""
import argparse, os, time, math, numpy as np, torch, h5py
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cond_model import ConditionedVIT
from dag_loss import DagHopLoss

NUM_CLASSES = 99


def maxnorm(X):
    m = X.max(axis=1, keepdims=True); m[m <= 0] = 1.0
    return (X / m).astype(np.float32)


def load_bundle(path, n=None, seed=0):
    with h5py.File(path, "r") as f:
        N = f["X"].shape[0]
        idx = np.arange(N)
        if n and n < N:
            idx = np.sort(np.random.default_rng(seed).choice(N, n, replace=False))
        X = maxnorm(np.asarray(f["X"][idx], np.float32))
        eg = np.asarray(f["eg"][idx], np.int64)
        if eg.min() >= 1:
            eg = eg - 1                          # 1..99 -> 0..98
        P = np.asarray(f["peaks"][idx], np.float32); PM = np.asarray(f["peak_mask"][idx], np.float32)
        C = np.asarray(f["cands"][idx], np.float32); CM = np.asarray(f["cand_mask"][idx], np.float32)
    return [torch.tensor(t) for t in (X, eg, P, PM, C, CM)]


def loader(bundle, bs, shuffle):
    return DataLoader(TensorDataset(*bundle), batch_size=bs, shuffle=shuffle,
                      drop_last=shuffle, num_workers=2, pin_memory=True)


def evaluate(model, dl, device, lossf):
    model.eval(); c1 = c5 = tot = 0; hop = 0.0
    with torch.no_grad():
        for X, y, P, PM, C, CM in dl:
            yd = y.to(device)
            out = model(X.to(device), P.to(device), PM.to(device), C.to(device), CM.to(device))
            top5 = out.topk(5, dim=1).indices.cpu()
            yv = y.view(-1, 1)
            c1 += (top5[:, :1] == yv).any(1).sum().item()
            c5 += (top5 == yv).any(1).sum().item()
            hop += lossf.mean_hop(out, yd) * len(y)
            tot += len(y)
    return 100.0 * c1 / tot, 100.0 * c5 / tot, hop / tot


def split_train_val(bundle, frac=0.1, seed=0):
    n = len(bundle[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = max(int(n * frac), 1)
    vi, ti = perm[:nv], perm[nv:]
    return [t[ti] for t in bundle], [t[vi] for t in bundle]


def _cos_warmup(step, total, warm=0.1):
    w = max(int(total * warm), 1)
    if step < w:
        return step / w
    p = (step - w) / max(total - w, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def train_stage(model, dl, total_steps, lr, device, lossf, val_dl=None,
                clip=1.0, eval_every=None):
    """Train for a FIXED optimizer-step budget (isolates volume from epoch count).
    cosine+warmup LR, grad clipping. If val_dl: early-stop on best synthetic-val Top-1
    (restore that checkpoint) -> avoids reporting a diverged final state."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cos_warmup(s, total_steps))
    eval_every = eval_every or max(total_steps // 6, 1)
    best_v, best_state = -1.0, None
    step = 0
    model.train()
    while step < total_steps:
        for X, y, P, PM, C, CM in dl:
            out = model(X.to(device), P.to(device), PM.to(device), C.to(device), CM.to(device))
            loss = lossf(out, y.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
            step += 1
            if val_dl is not None and step % eval_every == 0:
                v1, _, _ = evaluate(model, val_dl, device, lossf)
                if v1 > best_v:
                    best_v = v1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
            if step >= total_steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_v


def run(scale, conditioned, args, device, lam, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    uni = load_bundle(args.uniform, n=scale, seed=seed)
    ev = load_bundle(args.eval473)
    model = ConditionedVIT(spec_length=8501, patch_size=25, num_output=NUM_CLASSES,
                           embed_dim=128, depth=4, num_heads=4, use_physics_pe=True,
                           use_coordinate_channel=True,
                           cond_drop=0.3 if conditioned else 1.0,
                           n_cand_feat=uni[4].shape[-1]).to(device)
    lossf = DagHopLoss(args.adj, device, lam=lam, n=NUM_CLASSES)
    # early-stop on the REAL structure-disjoint pool (better proxy for real-473 transfer than
    # synthetic-val) if provided; else fall back to a synthetic-val split. Pool is disjoint
    # from the 473 test, so this is clean model-selection, not test leakage.
    realval_dl = loader(load_bundle(args.realval), args.bs, False) if args.realval else None
    t0 = time.time()
    # FIXED optimizer-step budget per stage (isolates data volume from epoch count).
    utr, uval = split_train_val(uni, frac=args.val_frac, seed=seed)
    v1 = realval_dl or loader(uval, args.bs, False)
    s1 = args.steps1 if args.steps1 > 0 else args.epochs1 * len(loader(utr, args.bs, True))
    train_stage(model, loader(utr, args.bs, True), s1, args.lr, device, lossf,
                val_dl=v1, clip=args.grad_clip)                                            # stage 1
    stage2 = (args.steps2 > 0 or args.epochs2 > 0) and os.path.exists(args.rruff)
    if stage2:                                                                            # stage 2
        rru = load_bundle(args.rruff, n=scale, seed=seed)
        rtr, rval = split_train_val(rru, frac=args.val_frac, seed=seed)
        v2 = realval_dl or loader(rval, args.bs, False)
        s2 = args.steps2 if args.steps2 > 0 else args.epochs2 * len(loader(rtr, args.bs, True))
        train_stage(model, loader(rtr, args.bs, True), s2, args.lr * 0.5, device, lossf,
                    val_dl=v2, clip=args.grad_clip)
    a1, a5, hop = evaluate(model, loader(ev, args.bs, False), device, lossf)
    tag = "uni->rruff" if stage2 else "uni-only"
    print(f"  [{'COND' if conditioned else 'BASE'}] scale={scale:>6} ({tag}) "
          f"473 Top-1={a1:.1f} Top-5={a5:.1f} hop={hop:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return a1, a5, hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", default="/work2/ALLOC3/UserE/cond_precompute/uniform_train.h5")
    ap.add_argument("--rruff", default="/work2/ALLOC3/UserE/cond_precompute/rruff_train.h5")
    ap.add_argument("--eval473", default="/work2/ALLOC3/UserE/cond_precompute/eval_473.h5")
    ap.add_argument("--scales", type=int, nargs="+", default=[100000, 250000, 500000])
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs1", type=int, default=3)
    ap.add_argument("--epochs2", type=int, default=3)
    ap.add_argument("--steps1", type=int, default=0, help="fixed step budget stage1 (>0 overrides epochs1)")
    ap.add_argument("--steps2", type=int, default=0, help="fixed step budget stage2 (>0 overrides epochs2)")
    ap.add_argument("--val_frac", type=float, default=0.1, help="synthetic held-out val (fallback)")
    ap.add_argument("--realval", default="", help="real structure-disjoint pool bundle for early-stop")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.2],
                    help="DAG hop-loss weights to sweep (0 = plain CE)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help=">1 seed -> controlled study (mean+/-std), at lams[0]")
    ap.add_argument("--adj", default="artifacts/topology_graphs/extinction_group_adjacency.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} scales={a.scales} lams={a.lams} "
          f"seeds={a.seeds} epochs1={a.epochs1} epochs2={a.epochs2} lr={a.lr}", flush=True)

    if len(a.seeds) > 1:                              # controlled scaling study: mean+/-std over seeds
        lam = a.lams[0]
        print(f"\nCONTROLLED STUDY (lam={lam}, {len(a.seeds)} seeds)\n"
              f"{'scale':>8} {'arm':>5} {'Top-1':>14} {'Top-5':>14} {'hop':>12}")
        agg = {}
        for s in a.scales:
            for cond in (False, True):
                rs = [run(s, cond, a, device, lam, seed) for seed in a.seeds]
                t1 = np.array([r[0] for r in rs]); t5 = np.array([r[1] for r in rs])
                hp = np.array([r[2] for r in rs])
                agg[(s, cond)] = (t1, t5, hp)
                print(f"{s:>8} {'COND' if cond else 'BASE':>5} "
                      f"{t1.mean():>7.1f}+/-{t1.std():<5.1f} {t5.mean():>7.1f}+/-{t5.std():<5.1f} "
                      f"{hp.mean():>6.2f}+/-{hp.std():<4.2f}", flush=True)
        print(f"\n{'scale':>8} {'dTop-1':>14} {'dTop-5':>14}")
        for s in a.scales:
            bt1, bt5, _ = agg[(s, False)]; ct1, ct5, _ = agg[(s, True)]
            print(f"{s:>8} {ct1.mean()-bt1.mean():>+7.1f} (b{bt1.std():.1f}/c{ct1.std():.1f}) "
                  f"{ct5.mean()-bt5.mean():>+7.1f} (b{bt5.std():.1f}/c{ct5.std():.1f})", flush=True)
        return

    print(f"\n{'lam':>5} {'scale':>8} {'BASE T1':>8} {'COND T1':>8} {'dT1':>6} | "
          f"{'BASE T5':>8} {'COND T5':>8} {'dT5':>6} | {'BASE hop':>8} {'COND hop':>8} {'dHop':>6}")
    for lam in a.lams:
        for s in a.scales:
            b1, b5, bh = run(s, False, a, device, lam)
            c1, c5, ch = run(s, True, a, device, lam)
            print(f"{lam:>5.2f} {s:>8} {b1:>8.1f} {c1:>8.1f} {c1-b1:>+6.1f} | "
                  f"{b5:>8.1f} {c5:>8.1f} {c5-b5:>+6.1f} | {bh:>8.2f} {ch:>8.2f} {ch-bh:>+6.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
