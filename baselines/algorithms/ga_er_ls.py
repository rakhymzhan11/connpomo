from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from baselines.algorithms.ga_er import (
    _fitness_pop, _rank_bucket, _build_pairs,
    _er, _perturb_if_clone, _swap_mut,
)
from baselines.common.local_search import local_search


@njit(cache=True)
def _ga_ls_run(npop: int, iters: int, cr: float, mr: float,
               ls_passes: int, elite_k: int,
               dist: np.ndarray, bus_conn: np.ndarray, seed: int,
               trace_evals: np.ndarray, trace_dists: np.ndarray) -> tuple:
    N = dist.shape[0]
    np.random.seed(seed)
    pen_max = N - 1


    pop = np.empty((npop, N), np.int32)
    for i in range(npop):
        for j in range(N):
            pop[i, j] = j
        for j in range(N - 1, 0, -1):
            k = np.random.randint(0, j + 1)
            tmp = pop[i, j]; pop[i, j] = pop[i, k]; pop[i, k] = tmp


    d_raw  = np.empty(npop, np.float64)
    q_raw  = np.empty(npop, np.int32)
    d_par  = np.empty(npop, np.float64)
    q_par  = np.empty(npop, np.int32)
    idx    = np.empty(npop, np.int32)
    par    = np.empty_like(pop)
    off    = np.empty_like(pop)
    c1     = np.empty(N, np.int32)
    c2     = np.empty(N, np.int32)
    d_off  = np.empty(npop, np.float64)
    q_off  = np.empty(npop, np.int32)
    D2     = np.empty(2 * npop, np.float64)
    Q2     = np.empty(2 * npop, np.int32)
    IDX2   = np.empty(2 * npop, np.int32)

    m         = npop // 2
    ring_buf  = np.empty(m, np.int32)
    top_pairs = np.empty((m, 2), np.int32)

    best_pen   = npop * N
    best_dist  = 1e300
    best_route = np.empty(N, np.int32)
    first_feas_eval = -1
    evals   = 0
    n_trace = 0
    max_trace = trace_evals.shape[0]

    for it in range(iters):


        _fitness_pop(pop, dist, bus_conn, d_raw, q_raw)
        _rank_bucket(q_raw, d_raw, idx, pen_max)


        for i in range(npop):
            src = idx[i]
            for j in range(N):
                par[i, j] = pop[src, j]
            d_par[i] = d_raw[src]
            q_par[i] = q_raw[src]


        for i in range(elite_k):
            p_i, d_i = local_search(par[i], dist, bus_conn, ls_passes)
            q_par[i] = p_i
            d_par[i] = d_i


            for j in range(N):
                pop[idx[i], j] = par[i, j]


        for i in range(npop):
            bp = q_par[i]; bd = d_par[i]
            if bp < best_pen or (bp == best_pen and bd < best_dist):
                best_pen = bp; best_dist = bd
                for k in range(N):
                    best_route[k] = par[i, k]
                if n_trace < max_trace:
                    trace_evals[n_trace] = evals
                    lex = bd if bp == 0 else 1e6 * bp + bd
                    trace_dists[n_trace] = lex
                    n_trace += 1
            if bp == 0 and first_feas_eval < 0:
                first_feas_eval = evals


        tp_written = _build_pairs(m, m, ring_buf, top_pairs)

        oi = 0
        for r in range(tp_written):
            a_rel = top_pairs[r, 0]
            b_rel = top_pairs[r, 1]
            p1 = par[a_rel]
            p2 = par[b_rel]

            if np.random.random() < cr:
                _er(c1, c2, p1, p2)
                _perturb_if_clone(c1, p1, p2)
                _perturb_if_clone(c2, p1, p2)
            else:
                for t in range(N):
                    c1[t] = p1[t]; c2[t] = p2[t]

            if np.random.random() < mr:
                _swap_mut(c1)
            if np.random.random() < mr:
                _swap_mut(c2)

            for t in range(N): off[oi, t] = c1[t]
            oi += 1
            for t in range(N): off[oi, t] = c2[t]
            oi += 1


        _fitness_pop(off, dist, bus_conn, d_off, q_off)
        evals += npop


        for i in range(npop):
            D2[i]        = d_par[i]
            Q2[i]        = q_par[i]
            D2[npop + i] = d_off[i]
            Q2[npop + i] = q_off[i]
        _rank_bucket(Q2, D2, IDX2, pen_max)

        for i in range(npop):
            k2 = IDX2[i]
            if k2 < npop:
                for t in range(N): pop[i, t] = par[k2, t]
            else:
                m2 = k2 - npop
                for t in range(N): pop[i, t] = off[m2, t]


        mb = IDX2[0]
        bp2 = Q2[mb]; bd2 = D2[mb]
        if bp2 < best_pen or (bp2 == best_pen and bd2 < best_dist):
            best_pen = bp2; best_dist = bd2
            if mb < npop:
                for t in range(N): best_route[t] = par[mb, t]
            else:
                m2 = mb - npop
                for t in range(N): best_route[t] = off[m2, t]
            if n_trace < max_trace:
                trace_evals[n_trace] = evals
                lex = bd2 if bp2 == 0 else 1e6 * bp2 + bd2
                trace_dists[n_trace] = lex
                n_trace += 1
        if bp2 == 0 and first_feas_eval < 0:
            first_feas_eval = evals

    return best_pen, best_dist, best_route, first_feas_eval, n_trace


def warmup(N: int = 14) -> None:
    dummy_dist = np.ones((N, N), dtype=np.float64)
    dummy_conn = np.ones((N, N), dtype=np.bool_)
    tr_e = np.zeros(500, dtype=np.int64)
    tr_d = np.full(500, 1e300, dtype=np.float64)
    _ga_ls_run(10, 2, 1.0, 0.1, 2, 1,
               dummy_dist, dummy_conn, 0, tr_e, tr_d)


def run(dist: np.ndarray,
        bus_conn: np.ndarray,
        budget: int = 20_000,
        seed: int = 0,
        np_size: int = 100,
        cr: float = 1.0,
        mr: float = 0.10,
        ls_passes: int = 5,
        ls_frac: float = 0.10,
        time_limit_s: float | None = None) -> dict:
    t0 = time.perf_counter()
    N = dist.shape[0]
    elite_k = max(1, math.ceil(ls_frac * np_size))


    evals_per_gen = np_size + elite_k * ls_passes * N
    iters_from_budget = max(1, budget // evals_per_gen)

    if time_limit_s is not None:


        _cal_a = 10
        _cal_b = 50
        _te = np.zeros(_cal_b * 4 + 10, dtype=np.int64)
        _td = np.full(_cal_b * 4 + 10, 1e300, dtype=np.float64)
        _t_a = time.perf_counter()
        _ga_ls_run(np_size, _cal_a, cr, mr, ls_passes, elite_k,
                   dist, bus_conn, seed, _te, _td)
        dt_a = time.perf_counter() - _t_a
        _t_b = time.perf_counter()
        _ga_ls_run(np_size, _cal_b, cr, mr, ls_passes, elite_k,
                   dist, bus_conn, seed, _te, _td)
        dt_b = time.perf_counter() - _t_b
        t_one_gen = max((dt_b - dt_a) / (_cal_b - _cal_a), 1e-9)
        remaining = time_limit_s - (time.perf_counter() - t0)
        iters_from_time = max(1, int(remaining / t_one_gen))
        iters = min(iters_from_time, iters_from_budget)
    else:
        iters = iters_from_budget

    max_trace = iters * 4 + 10
    trace_evals = np.zeros(max_trace, dtype=np.int64)
    trace_dists = np.full(max_trace, 1e300, dtype=np.float64)

    t_before_run = time.perf_counter()
    best_pen, best_dist, best_route, first_feas_eval, n_trace = _ga_ls_run(
        np_size, iters, cr, mr, ls_passes, elite_k,
        dist, bus_conn, seed,
        trace_evals, trace_dists,
    )
    wall_time_s = time.perf_counter() - t0
    total_evals = iters * evals_per_gen


    t_run = time.perf_counter() - t_before_run
    raw_evals = trace_evals[:n_trace].tolist()
    raw_lex   = trace_dists[:n_trace].tolist()
    trace = []
    for ev, lx in zip(raw_evals, raw_lex):
        t_s = (ev / max(total_evals, 1)) * t_run + (t_before_run - t0)
        if lx >= 1e6:
            pen = int(lx // 1e6)
            dist_val = lx - pen * 1e6
        else:
            pen = 0
            dist_val = lx
        trace.append((t_s, pen, dist_val))

    return {
        "best_penalty":   int(best_pen),
        "best_distance":  float(best_dist),
        "best_route":     best_route.tolist(),
        "evaluations":    total_evals,
        "first_feasible": int(first_feas_eval),
        "trace":          trace,
        "n_iterations":   iters,
        "wall_time_s":    wall_time_s,
        "time_in_ls_s":   0.0,
        "time_in_init_s": t_before_run - t0,
    }
