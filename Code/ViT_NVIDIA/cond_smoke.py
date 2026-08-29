"""
Conditioned-ViT plumbing/smoke test: verify the conditioning branch trains end-to-end and
that conditioning is actually used (conditioned vs unconditioned baseline on the same split).
NOT a science result (tiny data) — a wiring check.

Reads cond_smoke.npz (X, eg, peaks, peak_mask, cands, cand_mask). Trains a small ConditionedVIT
with conditioning ON vs OFF (cond_drop=1.0 → always null) and reports train/val EG top-1.
"""
import argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from cond_model import ConditionedVIT


def load(npz):
    d = np.load(npz)
    eg = torch.tensor(d["eg"]).long()
    if eg.min() >= 1:                 # synthetic labels are 1..99 -> shift to 0..98
        eg = eg - 1
    return (torch.tensor(d["X"]), eg, torch.tensor(d["peaks"]),
            torch.tensor(d["peak_mask"]), torch.tensor(d["cands"]), torch.tensor(d["cand_mask"]))


def run(npz, epochs=40, bs=64, lr=3e-4, num_classes=99, device="cuda", conditioned=True, seed=0):
    torch.manual_seed(seed)
    X, eg, P, PM, C, CM = load(npz)
    ds = TensorDataset(X, eg, P, PM, C, CM)
    ntr = int(0.8 * len(ds))
    tr, va = random_split(ds, [ntr, len(ds) - ntr], generator=torch.Generator().manual_seed(seed))
    dltr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    dlva = DataLoader(va, batch_size=bs)

    model = ConditionedVIT(spec_length=8501, patch_size=25, num_output=num_classes,
                           embed_dim=128, depth=4, num_heads=4,
                           cond_drop=0.3 if conditioned else 1.0,  # 1.0 => always null = unconditioned
                           n_cand_feat=C.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def evalacc(dl):
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for X, y, P, PM, C, CM in dl:
                X, y = X.to(device), y.to(device)
                out = model(X, P.to(device), PM.to(device), C.to(device), CM.to(device))
                cor += (out.argmax(1) == y).sum().item(); tot += len(y)
        return 100.0 * cor / max(tot, 1)

    for ep in range(epochs):
        model.train()
        for X, y, P, PM, C, CM in dltr:
            X, y = X.to(device), y.to(device)
            out = model(X, P.to(device), PM.to(device), C.to(device), CM.to(device))
            loss = lossf(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [{'COND' if conditioned else 'BASE'}] ep{ep:3d} "
                  f"loss={loss.item():.3f} train={evalacc(dltr):.1f}% val={evalacc(dlva):.1f}%",
                  flush=True)
    return evalacc(dlva)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)
    print("== conditioned ==")
    vc = run(a.npz, epochs=a.epochs, device=dev, conditioned=True)
    print("== baseline (conditioning forced null) ==")
    vb = run(a.npz, epochs=a.epochs, device=dev, conditioned=False)
    print(f"\nSMOKE: conditioned val={vc:.1f}%  baseline val={vb:.1f}%  delta={vc-vb:+.1f}")
