from __future__ import annotations

import time
import numpy as np
from numba import njit

from baselines.common.evaluation import eval_route


@njit(cache=True)
def _shuffle_i32(arr: np.ndarray) -> None:
    n = arr.shape[0]
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp


@njit(cache=True)
def _swap_mut(route: np.ndarray) -> None:
    n = route.shape[0]
    i = np.random.randint(0, n)
    j = i
    while j == i:
        j = np.random.randint(0, n)
    tmp = route[i]; route[i] = route[j]; route[j] = tmp


@njit(cache=True)
def _pick_unused(used: np.ndarray) -> int:
    n = used.shape[0]
    cnt = 0
    for i in range(n):
        if not used[i]:
            cnt += 1
    if cnt == 0:
        return -1
    r = np.random.randint(0, cnt)
    k = 0
    for i in range(n):
        if not used[i]:
            if k == r:
                return i
            k += 1
    return -1


@njit(cache=True)
def _rank_bucket(pen: np.ndarray, dist: np.ndarray,
                 idx_out: np.ndarray, pen_max: int) -> None:
    P = pen.shape[0]
    counts = np.zeros(pen_max + 1, np.int32)
    for i in range(P):
        v = pen[i] if pen[i] <= pen_max else pen_max
        counts[v] += 1

    offset = np.zeros(pen_max + 1, np.int32)
    s = 0
    for v in range(pen_max + 1):
        offset[v] = s
        s += counts[v]

    bucket_idx = np.empty(P, np.int32)
    write = offset.copy()
    for i in range(P):
        v = pen[i] if pen[i] <= pen_max else pen_max
        pos = write[v]
        bucket_idx[pos] = i
        write[v] = pos + 1


    for v in range(pen_max + 1):
        start = offset[v]
        end = start + counts[v]
        for i in range(start + 1, end):
            key = bucket_idx[i]
            j = i - 1
            while j >= start and dist[bucket_idx[j]] > dist[key]:
                bucket_idx[j + 1] = bucket_idx[j]
                j -= 1
            bucket_idx[j + 1] = key

    for i in range(P):
        idx_out[i] = bucket_idx[i]


@njit(cache=True)
def _fitness_pop(pop: np.ndarray,
                 dist: np.ndarray,
                 bus_conn: np.ndarray,
                 out_d: np.ndarray,
                 out_q: np.ndarray) -> None:
    P = pop.shape[0]
    N = pop.shape[1]
    for i in range(P):
        d = 0.0
        q = 0
        for t in range(N - 1):
            a = pop[i, t]
            b = pop[i, t + 1]
            d += dist[a, b]
            if not bus_conn[a, b]:
                q += 1
        out_d[i] = d
        out_q[i] = q


@njit(cache=True)
def _add_edge(edge_map: np.ndarray, deg: np.ndarray, a: int, b: int) -> None:
    if a == b:
        return

    dup = False
    for k in range(deg[a]):
        if edge_map[a, k] == b:
            dup = True
            break
    if not dup and deg[a] < 4:
        edge_map[a, deg[a]] = b
        deg[a] += 1

    dup = False
    for k in range(deg[b]):
        if edge_map[b, k] == a:
            dup = True
            break
    if not dup and deg[b] < 4:
        edge_map[b, deg[b]] = a
        deg[b] += 1


@njit(cache=True)
def _make_child(out: np.ndarray, p_start: int,
                edge_map: np.ndarray, deg: np.ndarray) -> None:
    n = edge_map.shape[0]
    used = np.zeros(n, np.uint8)

    current = p_start
    out[0] = current
    used[current] = 1

    for pos in range(1, n):

        for j in range(n):
            for k in range(deg[j]):
                if edge_map[j, k] == current:
                    for m in range(k, deg[j] - 1):
                        edge_map[j, m] = edge_map[j, m + 1]
                    edge_map[j, deg[j] - 1] = -1
                    deg[j] -= 1
                    break

        dcur = deg[current]
        if dcur > 0:
            min_deg = 127
            best = -1
            for k in range(dcur):
                g = edge_map[current, k]
                if g < 0:
                    continue
                if deg[g] < min_deg:
                    min_deg = deg[g]
                    best = g
            next_city = best
        else:
            next_city = _pick_unused(used)

        if next_city < 0:
            break
        out[pos] = next_city
        used[next_city] = 1
        current = next_city


@njit(cache=True)
def _er(c1: np.ndarray, c2: np.ndarray,
        p1: np.ndarray, p2: np.ndarray) -> None:
    n = p1.shape[0]
    edge_map = np.full((n, 4), -1, dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)

    for i in range(n):
        _add_edge(edge_map, deg, int(p1[i]), int(p1[(i + 1) % n]))
    for i in range(n):
        _add_edge(edge_map, deg, int(p2[i]), int(p2[(i + 1) % n]))

    _make_child(c1, int(p1[0]), edge_map.copy(), deg.copy())
    _make_child(c2, int(p2[0]), edge_map.copy(), deg.copy())


@njit(cache=True)
def _perturb_if_clone(child: np.ndarray,
                      p1: np.ndarray, p2: np.ndarray) -> None:
    n = child.shape[0]
    is_p1 = True
    is_p2 = True
    for i in range(n):
        if child[i] != p1[i]:
            is_p1 = False
        if child[i] != p2[i]:
            is_p2 = False
        if not is_p1 and not is_p2:
            break
    if is_p1 or is_p2:
        _swap_mut(child)


@njit(cache=True)
def _build_pairs(m: int, pairs_needed: int,
                 ring_buf: np.ndarray,
                 out_pairs: np.ndarray) -> int:
    for i in range(m):
        ring_buf[i] = i
    _shuffle_i32(ring_buf)
    count = 0
    for r in range(m):
        if count >= pairs_needed:
            break
        a = ring_buf[r]
        b = ring_buf[(r + 1) % m]
        if a == b:
            continue
        out_pairs[count, 0] = a
        out_pairs[count, 1] = b
        count += 1
    return count


@njit(cache=True)
def _ga_run(npop: int, iters: int, cr: float, mr: float,
            dist: np.ndarray, bus_conn: np.ndarray, seed: int,
            trace_evals: np.ndarray, trace_dists: np.ndarray) -> tuple:
    N = dist.shape[0]
    np.random.seed(seed)
    pen_max = N - 1


    pop = np.empty((npop, N), np.int32)
    base = np.arange(N, dtype=np.int32)
    for i in range(npop):
        for j in range(N):
            pop[i, j] = base[j]
        for j in range(N - 1, 0, -1):
            k = np.random.randint(0, j + 1)
            tmp = pop[i, j]; pop[i, j] = pop[i, k]; pop[i, k] = tmp


    d     = np.empty(npop, np.float64)
    q     = np.empty(npop, np.int32)
    idx   = np.empty(npop, np.int32)
    par   = np.empty_like(pop)
    off   = np.empty_like(pop)
    c1    = np.empty(N, np.int32)
    c2    = np.empty(N, np.int32)
    d_off = np.empty(npop, np.float64)
    q_off = np.empty(npop, np.int32)
    D2    = np.empty(2 * npop, np.float64)
    Q2    = np.empty(2 * npop, np.int32)
    IDX2  = np.empty(2 * npop, np.int32)

    m         = npop // 2
    ring_buf  = np.empty(m, np.int32)
    top_pairs = np.empty((m, 2), np.int32)

    best_pen  = npop * N
    best_dist = 1e300
    best_route = np.empty(N, np.int32)
    first_feas_eval = -1
    evals = 0
    n_trace = 0
    max_trace = trace_evals.shape[0]

    for it in range(iters):


        _fitness_pop(pop, dist, bus_conn, d, q)
        _rank_bucket(q, d, idx, pen_max)


        for i in range(npop):
            src = idx[i]
            for j in range(N):
                par[i, j] = pop[src, j]


        bp = q[idx[0]]; bd = d[idx[0]]
        if bp < best_pen or (bp == best_pen and bd < best_dist):
            best_pen = bp; best_dist = bd
            for k in range(N):
                best_route[k] = par[0, k]
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
            D2[i]        = d[idx[i]]
            Q2[i]        = q[idx[i]]
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
    tr_e = np.zeros(500, np.int64)
    tr_d = np.full(500, 1e300)
    _ga_run(10, 5, 1.0, 0.1, dummy_dist, dummy_conn, 0, tr_e, tr_d)


def run(dist: np.ndarray,
        bus_conn: np.ndarray,
        budget: int = 20_000,
        seed: int = 0,
        np_size: int = 100,
        cr: float = 1.0,
        mr: float = 0.10,
        time_limit_s: float | None = None) -> dict:
    t0 = time.perf_counter()


    if time_limit_s is not None:
        _cal_a = 20
        _cal_b = 200
        _te = np.zeros(_cal_b * 2 + 10, dtype=np.int64)
        _td = np.full(_cal_b * 2 + 10, 1e300, dtype=np.float64)
        _t_a = time.perf_counter()
        _ga_run(np_size, _cal_a, cr, mr, dist, bus_conn, seed, _te, _td)
        dt_a = time.perf_counter() - _t_a
        _t_b = time.perf_counter()
        _ga_run(np_size, _cal_b, cr, mr, dist, bus_conn, seed, _te, _td)
        dt_b = time.perf_counter() - _t_b
        t_one_gen = max((dt_b - dt_a) / (_cal_b - _cal_a), 1e-9)
        remaining = time_limit_s - (time.perf_counter() - t0)
        iters_from_time = max(1, int(remaining / t_one_gen))
        iters_from_budget = max(1, budget // np_size)
        iters = min(iters_from_time, iters_from_budget)
    else:
        iters = max(1, budget // np_size)


    max_trace = iters * 2 + 10
    trace_evals = np.zeros(max_trace, dtype=np.int64)
    trace_dists = np.full(max_trace, 1e300, dtype=np.float64)

    t_before_run = time.perf_counter()
    best_pen, best_dist, best_route, first_feas_eval, n_trace = _ga_run(
        np_size, iters, cr, mr,
        dist, bus_conn, seed,
        trace_evals, trace_dists,
    )
    wall_time_s = time.perf_counter() - t0


    t_run = time.perf_counter() - t_before_run
    raw_evals = trace_evals[:n_trace].tolist()
    raw_lex   = trace_dists[:n_trace].tolist()
    total_evals = np_size * iters
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
        "evaluations":    int(np_size * iters),
        "first_feasible": int(first_feas_eval),
        "trace":          trace,
        "n_iterations":   iters,
        "wall_time_s":    wall_time_s,
        "time_in_ls_s":   0.0,
        "time_in_init_s": t_before_run - t0,
    }
