import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import generate_batch
from model import ConnPOMO, arc_violations, tour_length

PENALTY = 2.0
BATCH = {14: 128, 20: 128, 30: 128, 50: 64, 100: 24}


def reward(tour, xy, conn):
    length = tour_length(tour, xy)
    viol = arc_violations(tour, conn)
    return -length - PENALTY * viol, length, viol == 0


def train_epoch(model, opt, a, rng, dev):
    model.train()
    K = min(a.K, a.N)
    steps = max(1, a.inst // a.batch)
    tot = np.zeros(3)
    for _ in range(steps):
        xy_np, conn_np = generate_batch(a.batch, a.N, a.hardness, rng)
        xy = torch.from_numpy(xy_np).to(dev).repeat_interleave(K, 0)
        conn = torch.from_numpy(conn_np).to(dev).repeat_interleave(K, 0)
        starts = torch.stack([torch.randperm(a.N)[:K] for _ in range(a.batch)]).reshape(-1).to(dev)
        tour, logp, _ = model(xy, conn, starts, decode_type="sample", use_pi=True)
        r, length, feas = reward(tour, xy, conn)
        rb = r.view(a.batch, K)
        adv = (rb - rb.mean(dim=1, keepdim=True)).reshape(-1).detach()
        loss = -(adv * logp).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += [float(r.mean()), float(feas.float().mean()), float(length.mean())]
    return tot / steps


def validate(model, valset, K, dev):
    model.eval()
    solved, lens = 0, []
    for xy_np, conn_np in valset:
        N = xy_np.shape[1]
        k = min(K, N)
        xy = torch.from_numpy(xy_np).to(dev).expand(k, N, 2).contiguous()
        conn = torch.from_numpy(conn_np).to(dev).expand(k, N, N).contiguous()
        with torch.no_grad():
            tour, _, _ = model(xy, conn, torch.arange(k, device=dev), decode_type="sample", use_pi=True)
            _, length, feas = reward(tour, xy, conn)
        if bool(feas.any()):
            solved += 1
            lens.append(float(length[feas].min()))
    return solved / len(valset), (float(np.mean(lens)) if lens else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--hardness", default="hard", choices=["easy", "medium", "hard"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--inst", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--val-inst", type=int, default=128)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    a.batch = a.batch or BATCH.get(a.N, 64)
    out = Path(a.out or f"runs/N{a.N}_{a.hardness}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = ConnPOMO().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    valset = [generate_batch(1, a.N, a.hardness, np.random.default_rng(900_000 + a.N + i))
              for i in range(a.val_inst)]
    start, best, best_ep, hist = 0, (-1.0, -1e18), 0, []
    resume = out / "resume.pt"
    if resume.exists() and not a.no_resume:
        st = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, best, best_ep, hist = st["epoch"], tuple(st["best"]), st["best_ep"], st["hist"]
        torch.set_rng_state(st["torch_rng"])
        rng.bit_generator.state = st["np_rng"]
        print(f"resuming after epoch {start}", flush=True)
    vs, vl = 0.0, float("nan")
    t0 = time.time()
    for ep in range(start + 1, a.epochs + 1):
        r, f, l = train_epoch(model, opt, a, rng, dev)
        if ep % a.val_every == 0 or ep == a.epochs:
            vs, vl = validate(model, valset, 2 * a.K, dev)
        score = (vs, -(vl if vl == vl else 1e9))
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), out / "best.pt")
        hist.append(dict(epoch=ep, reward=r, rollout_feasible=f, length=l,
                         val_solved=vs, val_length=vl, elapsed_s=time.time() - t0))
        print(f"epoch {ep:3d}  reward {r:.4f}  rollout feasible {f:.3f}  "
              f"val solved {vs:.3f}  val length {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, best=best,
                        best_ep=best_ep, hist=hist, torch_rng=torch.get_rng_state(),
                        np_rng=rng.bit_generator.state), resume)
        if a.patience and ep >= a.min_epochs and ep - best_ep >= a.patience:
            print(f"stopping at epoch {ep}, no new best since epoch {best_ep}", flush=True)
            break
    torch.save(model.state_dict(), out / "final.pt")
    json.dump(vars(a) | dict(total_s=time.time() - t0), open(out / "config.json", "w"), indent=1)
    print(f"best epoch {best_ep}: val solved {best[0]:.3f}, length {-best[1]:.4f}")


if __name__ == "__main__":
    main()
