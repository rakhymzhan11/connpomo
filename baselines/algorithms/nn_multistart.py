from __future__ import annotations

import time

import numpy as np
from numba import njit

from baselines.common.evaluation import eval_route, lex_better
from baselines.common.local_search import local_search


@njit(cache=True)
def _nn_construct(start: int,
                  dist: np.ndarray,
                  bus_conn: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    route = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    route[0] = start
    visited[start] = True

    for step in range(1, n):
        cur = route[step - 1]
        best_feas_d  = 1e300
        best_feas_j  = -1
        best_any_d   = 1e300
        best_any_j   = -1

        for j in range(n):
            if visited[j]:
                continue
            d = dist[cur, j]
            if d < best_any_d:
                best_any_d = d
                best_any_j = j
            if bus_conn[cur, j] and d < best_feas_d:
                best_feas_d = d
                best_feas_j = j

        nxt = best_feas_j if best_feas_j >= 0 else best_any_j
        route[step] = nxt
        visited[nxt] = True

    return route


def run(dist: np.ndarray,
        bus_conn: np.ndarray,
        budget: int = 20_000,
        seed: int = 0,
        ls_passes: int = 50,
        use_ls: bool = True,
        time_limit_s: float | None = None) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = dist.shape[0]

    best_p = 10 ** 9
    best_d = 1e300
    best_route = np.arange(n, dtype=np.int32)
    evals = 0
    first_feasible = -1
    trace = []
    n_iters = 0
    time_in_ls_s = 0.0


    start_order = list(range(n))
    rng.shuffle(start_order)

    restart_idx = 0
    while evals < budget:
        if time_limit_s is not None and time.perf_counter() - t0 >= time_limit_s:
            break

        start = start_order[restart_idx % n]
        restart_idx += 1


        route = _nn_construct(start, dist, bus_conn)
        p, d = eval_route(route, dist, bus_conn)
        evals += 1

        if first_feasible == -1 and p == 0:
            first_feasible = evals

        if lex_better(p, d, best_p, best_d):
            best_p, best_d = p, d
            best_route = route.copy()
            trace.append((time.perf_counter() - t0, int(best_p), float(best_d)))


        if use_ls and evals < budget:
            passes = min(ls_passes, (budget - evals) // max(1, n))
            passes = max(1, passes)
            route_ls = route.copy()
            _ls0 = time.perf_counter()
            p_ls, d_ls = local_search(route_ls, dist, bus_conn, passes)
            time_in_ls_s += time.perf_counter() - _ls0
            evals = min(evals + passes * n, budget)

            if first_feasible == -1 and p_ls == 0:
                first_feasible = evals
            if lex_better(p_ls, d_ls, best_p, best_d):
                best_p, best_d = p_ls, d_ls
                best_route = route_ls.copy()
                trace.append((time.perf_counter() - t0, int(best_p), float(best_d)))

        n_iters += 1

    time_in_init_s = 0.0

    return {
        "best_penalty":   int(best_p),
        "best_distance":  float(best_d),
        "best_route":     best_route.tolist(),
        "evaluations":    evals,
        "first_feasible": first_feasible,
        "trace":          trace,
        "n_iterations":   n_iters,
        "wall_time_s":    time.perf_counter() - t0,
        "time_in_ls_s":   time_in_ls_s,
        "time_in_init_s": time_in_init_s,
    }
