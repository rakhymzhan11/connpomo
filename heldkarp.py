import sys

import numpy as np
from numba import njit

from data import load_instances


@njit(cache=True)
def held_karp(dist, conn):
    N = dist.shape[0]
    FULL = (1 << N) - 1
    INF = 1e18
    dp = np.full((1 << N, N), INF)
    for s in range(N):
        dp[1 << s, s] = 0.0
    for mask in range(1, 1 << N):
        for u in range(N):
            if dp[mask, u] >= INF or not (mask >> u) & 1:
                continue
            for v in range(N):
                if (mask >> v) & 1 or conn[u, v] == 0:
                    continue
                c = dp[mask, u] + dist[u, v]
                if c < dp[mask | (1 << v), v]:
                    dp[mask | (1 << v), v] = c
    best = INF
    for v in range(N):
        if dp[FULL, v] < best:
            best = dp[FULL, v]
    return best


if __name__ == "__main__":
    for path in sys.argv[1:]:
        opt = [held_karp(d, c.astype(np.int32)) for _, c, d in load_instances(path)]
        print(f"{path}: optimum mean {np.mean(opt):.4f}  " + " ".join(f"{o:.3f}" for o in opt))
