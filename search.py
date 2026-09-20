import numpy as np
from numba import njit


@njit(cache=True)
def feasible(tour, conn):
    for t in range(tour.shape[0] - 1):
        if conn[tour[t], tour[t + 1]] == 0:
            return False
    return True


@njit(cache=True)
def warnsdorff(conn, dist, start, max_steps):
    N = conn.shape[0]
    visited = np.zeros(N, dtype=np.bool_)
    path = np.full(N, -1, dtype=np.int64)
    tried = np.zeros((N, N), dtype=np.bool_)
    visited[start] = True
    path[0] = start
    depth = 1
    steps = 0
    while depth < N and steps < max_steps:
        steps += 1
        cur = path[depth - 1]
        best_j = -1
        best_v = -1e30
        for j in range(N):
            if conn[cur, j] == 0 or visited[j] or tried[depth - 1, j]:
                continue
            deg = 0
            for k in range(N):
                if conn[j, k] != 0 and not visited[k] and k != j:
                    deg += 1
            v = -1.0 * deg - 0.5 * dist[cur, j]
            if v > best_v:
                best_v = v
                best_j = j
        if best_j < 0:
            if depth == 1:
                return path, False
            last = path[depth - 1]
            visited[last] = False
            for k in range(N):
                tried[depth - 1, k] = False
            path[depth - 1] = -1
            depth -= 1
            tried[depth - 1, last] = True
            continue
        tried[depth - 1, best_j] = True
        path[depth] = best_j
        visited[best_j] = True
        depth += 1
    return path, depth >= N


def warmup():
    c = np.ones((4, 4), dtype=np.int32)
    np.fill_diagonal(c, 0)
    d = np.random.default_rng(0).random((4, 4))
    warnsdorff(c, d, 0, 100)
    feasible(np.arange(4), c)
