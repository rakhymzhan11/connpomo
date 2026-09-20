from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def eval_route(route: np.ndarray,
               dist: np.ndarray,
               bus_conn: np.ndarray) -> tuple[int, float]:
    n = route.shape[0]
    penalty = 0
    distance = 0.0
    for t in range(n - 1):
        a = route[t]
        b = route[t + 1]
        if not bus_conn[a, b]:
            penalty += 1
        distance += dist[a, b]
    return penalty, distance


@njit(cache=True)
def lex_better(p1: int, d1: float, p2: int, d2: float) -> bool:
    if p1 < p2:
        return True
    if p1 == p2 and d1 < d2:
        return True
    return False


@njit(cache=True)
def lex_better_or_equal(p1: int, d1: float, p2: int, d2: float) -> bool:
    if p1 < p2:
        return True
    if p1 == p2 and d1 <= d2:
        return True
    return False


@njit(cache=True)
def eval_population(pop: np.ndarray,
                    dist: np.ndarray,
                    bus_conn: np.ndarray,
                    penalties: np.ndarray,
                    distances: np.ndarray) -> None:
    np_size = pop.shape[0]
    for i in range(np_size):
        p, d = eval_route(pop[i], dist, bus_conn)
        penalties[i] = p
        distances[i] = d


def eval_route_py(route, D: np.ndarray, B: np.ndarray) -> tuple[int, float]:
    route = np.asarray(route, dtype=np.int32)
    return eval_route(route, D, B)


def is_feasible(route, B: np.ndarray) -> bool:
    p, _ = eval_route_py(route, np.zeros_like(B, dtype=np.float64), B)
    return p == 0


def warmup(N: int = 14) -> None:
    dummy_route = np.arange(N, dtype=np.int32)
    dummy_dist  = np.ones((N, N), dtype=np.float64)
    dummy_conn  = np.ones((N, N), dtype=np.bool_)
    eval_route(dummy_route, dummy_dist, dummy_conn)
    lex_better(0, 1.0, 0, 2.0)
    lex_better_or_equal(0, 1.0, 0, 1.0)
    pop = np.tile(dummy_route, (4, 1))
    pens = np.zeros(4, dtype=np.int32)
    dists = np.zeros(4, dtype=np.float64)
    eval_population(pop, dummy_dist, dummy_conn, pens, dists)
