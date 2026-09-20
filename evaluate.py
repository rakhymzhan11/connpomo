import argparse
import json
import time

import numpy as np
import torch

from data import dihedral, load_instances
from model import load_model


def solve(model, xy, conn, dist, use_pi=True, decode="sample"):
    N = xy.shape[0]
    K = min(N, 64)
    CN = torch.tensor(conn).unsqueeze(0).expand(K, N, N).contiguous()
    best = None
    with torch.no_grad():
        for aug in dihedral(xy):
            XY = torch.tensor(np.ascontiguousarray(aug)).unsqueeze(0).expand(K, N, 2).contiguous()
            for rep in range(2):
                st = torch.arange(K) % N if rep == 0 else torch.randint(0, N, (K,))
                tours, _, _ = model(XY, CN, st, decode_type=decode, use_pi=use_pi)
                for t in tours.numpy():
                    if len(set(t.tolist())) == N and all(conn[t[i], t[i + 1]] > 0 for i in range(N - 1)):
                        L = float(sum(dist[t[i], t[i + 1]] for i in range(N - 1)))
                        best = L if best is None else min(best, L)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--instances", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    insts = load_instances(a.instances)
    N = insts[0][0].shape[0]
    model = load_model(a.ckpt)
    seed = a.seed if a.seed is not None else 1007 + N
    torch.manual_seed(seed)
    per = []
    t0 = time.perf_counter()
    for xy, conn, dist in insts:
        per.append(solve(model, xy, conn, dist, not a.no_mask, "greedy" if a.greedy else "sample"))
    sec = (time.perf_counter() - t0) / len(insts)
    lens = [x for x in per if x is not None]
    mean = float(np.mean(lens)) if lens else None
    print(f"{a.instances}: solved {len(lens)}/{len(insts)}  mean length "
          f"{mean if mean is None else round(mean, 4)}  {sec:.3f} s/instance")
    if a.out:
        json.dump(dict(ckpt=a.ckpt, instances=a.instances, seed=seed, mask=not a.no_mask,
                       decode="greedy" if a.greedy else "sample", solved=len(lens),
                       n=len(insts), length=mean, sec=sec, per_instance=per),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
