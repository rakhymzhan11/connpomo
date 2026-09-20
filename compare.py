import argparse
import json
import time

import numpy as np

import search
from baselines.algorithms.acs import run as acs
from baselines.algorithms.ga_er import run as ga
from baselines.algorithms.ga_er_ls import run as ga_ls
from baselines.algorithms.nn_multistart import run as nn
from data import load_instances

METHODS = {"Warnsdorff": None,
        "GA(ER)": (ga, dict(np_size=100, cr=0.8, mr=0.1)),
        "GA(ER)+LS": (ga_ls, dict(np_size=100, cr=0.8, mr=0.1)),
        "NN multi-start": (nn, dict(use_ls=True)),
        "ACS": (acs, dict(n_ants=50, use_ls=False)),
        "ACS+LS": (acs, dict(n_ants=50, use_ls=True))}
def run_search(conn, dist):
    N = conn.shape[0]
    best = None
    for s in range(min(N, 100)):
        path, ok = search.warnsdorff(conn, dist, s, 200_000)
        if ok and search.feasible(path, conn):
            L = float(sum(dist[a, b] for a, b in zip(path[:-1], path[1:])))
            best = L if best is None else min(best, L)
    return best


def run_meta(fn, kw, conn, dist, restarts):
    N = conn.shape[0]
    best = None
    for s in range(restarts):
        r = fn(dist, conn, budget=100 * N, seed=s, **kw)
        if r["best_penalty"] == 0:
            best = r["best_distance"] if best is None else min(best, r["best_distance"])
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--methods", default=None)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    search.warmup()
    insts = [(c.astype(np.int32), d) for _, c, d in load_instances(a.instances)]
    names = a.methods.split(",") if a.methods else list(METHODS)
    out = {}
    for name in names:
        per = []
        t0 = time.perf_counter()
        for conn, dist in insts:
            if METHODS[name] is None:
                per.append(run_search(conn, dist))
            else:
                per.append(run_meta(*METHODS[name], conn, dist, a.restarts))
        sec = (time.perf_counter() - t0) / len(insts)
        lens = [x for x in per if x is not None]
        mean = float(np.mean(lens)) if lens else None
        out[name] = dict(solved=len(lens), n=len(insts), length=mean, sec=sec, per_instance=per)
        print(f"{name:<15} solved {len(lens):>3}/{len(insts)}  mean length "
              f"{mean if mean is None else round(mean, 4)}  {sec:.3f} s/instance", flush=True)
    if a.out:
        json.dump(dict(instances=a.instances, results=out),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
