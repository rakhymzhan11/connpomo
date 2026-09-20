from __future__ import annotations

import numpy as np
from numba import njit

from baselines.common.evaluation import eval_route, lex_better


@njit(cache=True)
def _reverse_segment(route: np.ndarray, i: int, k: int) -> None:
    while i < k:
        tmp = route[i]
        route[i] = route[k]
        route[k] = tmp
        i += 1
        k -= 1


@njit(cache=True)
def two_opt(route: np.ndarray,
            dist: np.ndarray,
            bus_conn: np.ndarray,
            max_passes: int = 100) -> tuple[int, float]:
    n = route.shape[0]
    cur_p, cur_d = eval_route(route, dist, bus_conn)

    for _ in range(max_passes):
        best_dp = 0
        best_dd = 0.0
        best_i  = -1
        best_k  = -1

        for i in range(n - 1):
            for k in range(i + 1, n):

                if i == 0 and k == n - 1:
                    continue


                dp = 0

                if i > 0:
                    a, b = route[i - 1], route[i]
                    if not bus_conn[a, b]:
                        dp += 1

                    a2, b2 = route[i - 1], route[k]
                    if not bus_conn[a2, b2]:
                        dp -= 1


                if k < n - 1:
                    a, b = route[k], route[k + 1]
                    if not bus_conn[a, b]:
                        dp += 1

                    a2, b2 = route[i], route[k + 1]
                    if not bus_conn[a2, b2]:
                        dp -= 1


                dd = 0.0
                if i > 0:
                    dd -= dist[route[i - 1], route[i]]
                    dd += dist[route[i - 1], route[k]]
                if k < n - 1:
                    dd -= dist[route[k], route[k + 1]]
                    dd += dist[route[i], route[k + 1]]


                new_p = cur_p - dp
                new_d = cur_d + dd
                candidate_p = cur_p - best_dp
                candidate_d = cur_d + best_dd


                if best_i == -1:
                    if lex_better(new_p, new_d, cur_p, cur_d):
                        best_dp = dp
                        best_dd = dd
                        best_i  = i
                        best_k  = k
                else:
                    if lex_better(new_p, new_d, candidate_p, candidate_d):
                        best_dp = dp
                        best_dd = dd
                        best_i  = i
                        best_k  = k

        if best_i == -1:
            break

        _reverse_segment(route, best_i, best_k)
        cur_p -= best_dp
        cur_d += best_dd

    return cur_p, cur_d


@njit(cache=True)
def or_opt(route: np.ndarray,
           dist: np.ndarray,
           bus_conn: np.ndarray,
           chain_len: int = 1,
           max_passes: int = 100) -> tuple[int, float]:
    n = route.shape[0]
    if n <= chain_len + 1:
        return eval_route(route, dist, bus_conn)

    cur_p, cur_d = eval_route(route, dist, bus_conn)
    buf = np.empty(chain_len, dtype=np.int32)

    for _ in range(max_passes):
        best_dp = 0
        best_dd = 0.0
        best_src = -1
        best_dst = -1


        for i in range(n - chain_len + 1):
            i_end = i + chain_len - 1


            dp_remove = 0
            dd_remove = 0.0
            if i > 0:
                a, b = route[i - 1], route[i]
                if not bus_conn[a, b]:
                    dp_remove += 1
                dd_remove -= dist[a, b]
            if i_end < n - 1:
                a, b = route[i_end], route[i_end + 1]
                if not bus_conn[a, b]:
                    dp_remove += 1
                dd_remove -= dist[a, b]


            dp_bridge = 0
            dd_bridge = 0.0
            if i > 0 and i_end < n - 1:
                a, b = route[i - 1], route[i_end + 1]
                if not bus_conn[a, b]:
                    dp_bridge -= 1
                dd_bridge += dist[a, b]


            for j in range(-1, n - chain_len):


                if j == i - 1:
                    continue
                if i <= j <= i_end:
                    continue


                left_node  = route[j] if j >= 0 else -1


                if j < i - 1:
                    right_node = route[j + 1]
                elif j >= i_end + 1:
                    right_node = route[j + 1] if j + 1 < n else -1
                else:
                    continue

                chain_left  = route[i]
                chain_right = route[i_end]


                dp_insert = 0
                dd_insert = 0.0
                if left_node >= 0:
                    if not bus_conn[left_node, chain_left]:
                        dp_insert -= 1
                    dd_insert += dist[left_node, chain_left]
                if right_node >= 0:
                    if not bus_conn[chain_right, right_node]:
                        dp_insert -= 1
                    dd_insert += dist[chain_right, right_node]

                if left_node >= 0 and right_node >= 0:
                    if not bus_conn[left_node, right_node]:
                        dp_insert += 1
                    dd_insert -= dist[left_node, right_node]

                total_dp = dp_remove + dp_bridge + dp_insert
                total_dd = dd_remove + dd_bridge + dd_insert

                new_p = cur_p - total_dp
                new_d = cur_d + total_dd

                if best_src == -1:
                    if lex_better(new_p, new_d, cur_p, cur_d):
                        best_dp  = total_dp
                        best_dd  = total_dd
                        best_src = i
                        best_dst = j
                else:
                    if lex_better(new_p, new_d, cur_p - best_dp, cur_d + best_dd):
                        best_dp  = total_dp
                        best_dd  = total_dd
                        best_src = i
                        best_dst = j

        if best_src == -1:
            break


        for c in range(chain_len):
            buf[c] = route[best_src + c]


        tmp = np.empty(n, dtype=np.int32)
        w = 0
        for idx in range(n):
            if best_src <= idx <= best_src + chain_len - 1:
                continue
            tmp[w] = route[idx]
            w += 1


        if best_dst < best_src:
            ins = best_dst + 1
        else:
            ins = best_dst - chain_len + 1


        final = np.empty(n, dtype=np.int32)
        for idx in range(ins):
            final[idx] = tmp[idx]
        for c in range(chain_len):
            final[ins + c] = buf[c]
        for idx in range(ins, n - chain_len):
            final[ins + chain_len + idx - ins] = tmp[idx]

        for idx in range(n):
            route[idx] = final[idx]

        cur_p -= best_dp
        cur_d += best_dd

    return cur_p, cur_d


@njit(cache=True)
def local_search(route: np.ndarray,
                 dist: np.ndarray,
                 bus_conn: np.ndarray,
                 max_passes: int = 50) -> tuple[int, float]:
    prev_p = 10 ** 9
    prev_d = 1e300
    cur_p, cur_d = eval_route(route, dist, bus_conn)

    while lex_better(cur_p, cur_d, prev_p, prev_d):
        prev_p, prev_d = cur_p, cur_d
        cur_p, cur_d = two_opt(route, dist, bus_conn, max_passes)
        cur_p, cur_d = or_opt(route, dist, bus_conn, 1, max_passes)
        cur_p, cur_d = or_opt(route, dist, bus_conn, 2, max_passes)
        cur_p, cur_d = or_opt(route, dist, bus_conn, 3, max_passes)

    return cur_p, cur_d


def warmup(N: int = 14) -> None:
    route = np.arange(N, dtype=np.int32)
    dist  = np.ones((N, N), dtype=np.float64)
    conn  = np.ones((N, N), dtype=np.bool_)
    two_opt(route.copy(), dist, conn, 2)
    or_opt(route.copy(), dist, conn, 1, 2)
    or_opt(route.copy(), dist, conn, 2, 2)
    or_opt(route.copy(), dist, conn, 3, 2)
    local_search(route.copy(), dist, conn, 2)
