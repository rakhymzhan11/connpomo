from __future__ import annotations

import time

import numpy as np
from numba import njit

from baselines.common.evaluation import eval_route, lex_better
from baselines.common.local_search import local_search


@njit(cache=True)
def _acs_construct(tau: np.ndarray,
                   eta: np.ndarray,
                   bus_conn: np.ndarray,
                   alpha: float,
                   beta: float,
                   q0: float,
                   rho_local: float,
                   tau0: float,
                   k_candidates: int,
                   rng_state: np.ndarray) -> np.ndarray:
    n = tau.shape[0]

    def _lcg(state: np.ndarray) -> np.float64:
        state[0] = state[0] * np.int64(6364136223846793005) + np.int64(1442695040888963407)
        return np.float64((state[0] >> 11) & 0x1FFFFFFFFFFFFF) / np.float64(0x1FFFFFFFFFFFFF)

    def _lcg_int(state: np.ndarray, m: int) -> int:
        state[0] = state[0] * np.int64(6364136223846793005) + np.int64(1442695040888963407)
        return int((state[0] >> 33) & 0x7FFFFFFF) % m

    route = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    start = _lcg_int(rng_state, n)
    route[0] = start
    visited[start] = True

    for step in range(1, n):
        cur = route[step - 1]


        cand = np.empty(n, dtype=np.int32)
        cand_eta = np.empty(n, dtype=np.float64)
        cand_cnt = 0
        for j in range(n):
            if visited[j]:
                continue
            if bus_conn[cur, j] and eta[cur, j] > 0.0:
                cand[cand_cnt] = j
                cand_eta[cand_cnt] = eta[cur, j]
                cand_cnt += 1


        if cand_cnt == 0:

            for j in range(n):
                if not visited[j]:
                    cand[cand_cnt] = j
                    cand_eta[cand_cnt] = eta[cur, j] if eta[cur, j] > 0.0 else 1e-6
                    cand_cnt += 1


        k = min(k_candidates, cand_cnt)
        for a in range(k):
            best_ci = a
            for b in range(a + 1, cand_cnt):
                if cand_eta[b] > cand_eta[best_ci]:
                    best_ci = b
            if best_ci != a:
                tmp_j = cand[a]; cand[a] = cand[best_ci]; cand[best_ci] = tmp_j
                tmp_e = cand_eta[a]; cand_eta[a] = cand_eta[best_ci]; cand_eta[best_ci] = tmp_e

        q = _lcg(rng_state)

        if q <= q0:

            best_val = -1.0
            best_j = cand[0]
            for ci in range(k):
                j = cand[ci]
                val = (tau[cur, j] ** alpha) * (cand_eta[ci] ** beta)
                if val > best_val:
                    best_val = val
                    best_j = j
            nxt = best_j
        else:

            total = 0.0
            probs = np.empty(k, dtype=np.float64)
            for ci in range(k):
                j = cand[ci]
                w = (tau[cur, j] ** alpha) * (cand_eta[ci] ** beta)
                probs[ci] = w
                total += w

            if total <= 0.0:
                nxt = cand[_lcg_int(rng_state, k)]
            else:
                r = _lcg(rng_state) * total
                acc = 0.0
                nxt = cand[0]
                for ci in range(k):
                    acc += probs[ci]
                    if acc >= r:
                        nxt = cand[ci]
                        break

        route[step] = nxt
        visited[nxt] = True


        tau[cur, nxt] = (1.0 - rho_local) * tau[cur, nxt] + rho_local * tau0
        tau[nxt, cur] = tau[cur, nxt]

    return route


@njit(cache=True)
def _acs_global_update(tau: np.ndarray,
                       best_route: np.ndarray,
                       best_dist: float,
                       rho: float,
                       Q: float) -> None:
    n = tau.shape[0]


    for i in range(n):
        for j in range(n):
            tau[i, j] *= (1.0 - rho)


    if best_dist > 0.0:
        dep = Q / best_dist
        for t in range(n - 1):
            a = best_route[t]
            b = best_route[t + 1]
            tau[a, b] += dep
            tau[b, a] += dep


def run(dist: np.ndarray,
        bus_conn: np.ndarray,
        budget: int = 20_000,
        seed: int = 0,
        n_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 3.0,
        q0: float = 0.9,
        rho: float = 0.1,
        rho_local: float = 0.1,
        Q: float = 1.0,
        k_candidates: int = 10,
        use_ls: bool = True,
        ls_passes: int = 20,
        time_limit_s: float | None = None) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    lcg_state = np.array([rng.integers(1, 2**62)], dtype=np.int64)

    time_in_ls_s = 0.0


    eta = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j and bus_conn[i, j]:
                eta[i, j] = 1.0 / (dist[i, j] + 1e-12)


    nn_dist = 0.0
    visited = [False] * n
    cur = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d = 1e300
        best_j = -1
        for j in range(n):
            if not visited[j] and bus_conn[cur, j] and dist[cur, j] < best_d:
                best_d = dist[cur, j]
                best_j = j
        if best_j == -1:
            for j in range(n):
                if not visited[j] and dist[cur, j] < best_d:
                    best_d = dist[cur, j]
                    best_j = j
        nn_dist += best_d
        cur = best_j
        visited[cur] = True
    tau0 = 1.0 / (n * max(nn_dist, 1e-6))
    time_in_init_s = time.perf_counter() - t0

    tau = np.full((n, n), tau0, dtype=np.float64)
    np.fill_diagonal(tau, 0.0)

    best_p = 10 ** 9
    best_d_global = 1e300
    best_route = np.arange(n, dtype=np.int32)
    evals = 0
    first_feasible = -1
    trace = []
    colony_iters_done = 0

    n_iters = max(1, budget // n_ants)

    for _it in range(n_iters):
        if evals >= budget:
            break
        if time_limit_s is not None and time.perf_counter() - t0 >= time_limit_s:
            break

        iter_best_p = 10 ** 9
        iter_best_d = 1e300
        iter_best_route = np.arange(n, dtype=np.int32)

        for _ant in range(n_ants):
            if evals >= budget:
                break

            route = _acs_construct(tau, eta, bus_conn,
                                   alpha, beta, q0, rho_local, tau0,
                                   k_candidates, lcg_state)
            p, d = eval_route(route, dist, bus_conn)
            evals += 1

            if use_ls and evals < budget:
                passes = min(ls_passes, (budget - evals) // max(1, n))
                passes = max(1, passes)
                _ls0 = time.perf_counter()
                p, d = local_search(route, dist, bus_conn, passes)
                time_in_ls_s += time.perf_counter() - _ls0
                evals = min(evals + passes * n, budget)

            if first_feasible == -1 and p == 0:
                first_feasible = evals

            if lex_better(p, d, iter_best_p, iter_best_d):
                iter_best_p, iter_best_d = p, d
                iter_best_route = route.copy()

        colony_iters_done += 1


        if iter_best_d < 1e299:
            _acs_global_update(tau, iter_best_route, iter_best_d, rho, Q)

        if lex_better(iter_best_p, iter_best_d, best_p, best_d_global):
            best_p, best_d_global = iter_best_p, iter_best_d
            best_route = iter_best_route.copy()
            trace.append((time.perf_counter() - t0, int(best_p), float(best_d_global)))

    return {
        "best_penalty":   int(best_p),
        "best_distance":  float(best_d_global),
        "best_route":     best_route.tolist(),
        "evaluations":    evals,
        "first_feasible": first_feasible,
        "trace":          trace,
        "n_iterations":   colony_iters_done,
        "wall_time_s":    time.perf_counter() - t0,
        "time_in_ls_s":   time_in_ls_s,
        "time_in_init_s": time_in_init_s,
    }
