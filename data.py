import math
import sys
from pathlib import Path

import numpy as np

DENSITY = {"easy": 0.50, "medium": 0.25, "hard": 0.10}
GRID = [(N, h) for N in (14, 20, 30, 50) for h in ("easy", "medium", "hard")] + [(100, "hard")]


def generate_instance(N, hardness, rng):
    xy = rng.random((N, 2), dtype=np.float64).astype(np.float32)
    perm = rng.permutation(N)
    conn = np.zeros((N, N), dtype=np.float32)
    for k in range(N - 1):
        u, v = int(perm[k]), int(perm[k + 1])
        conn[u, v] = 1.0
        conn[v, u] = 1.0
    d = DENSITY[hardness]
    for i in range(N):
        for j in range(i + 1, N):
            if conn[i, j] == 0 and rng.random() < d:
                conn[i, j] = 1.0
                conn[j, i] = 1.0
    return xy, conn


def generate_batch(B, N, hardness, rng):
    xy = np.empty((B, N, 2), dtype=np.float32)
    conn = np.empty((B, N, N), dtype=np.float32)
    for b in range(B):
        xy[b], conn[b] = generate_instance(N, hardness, rng)
    return xy, conn


def euclid(xy):
    xy = np.asarray(xy, np.float32)
    return np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1)).astype(np.float64)


def unit_square(latlon):
    lat, lon = latlon[:, 0], latlon[:, 1]
    y = (lat - lat.mean()) * 110.574
    x = (lon - lon.mean()) * 111.320 * math.cos(math.radians(lat.mean()))
    s = max(x.max() - x.min(), y.max() - y.min())
    x = (x - x.min()) / s
    y = (y - y.min()) / s
    x += (1 - x.max()) / 2
    y += (1 - y.max()) / 2
    return np.stack([x, y], -1).astype(np.float32)


def dihedral(v):
    x, y = v[..., 0:1], v[..., 1:2]
    return [np.concatenate(t, -1) for t in
            ((x, y), (y, x), (1 - x, y), (y, 1 - x),
             (x, 1 - y), (1 - y, x), (1 - x, 1 - y), (1 - y, 1 - x))]


def load_instances(path):
    d = np.load(path)
    xy = d["xy"].astype(np.float32)
    conn = (d["conn"] > 0).astype(np.float32)
    dist = d["dist"].astype(np.float64)
    return [(xy[i], conn[i], dist[i]) for i in range(len(xy))]


def write_grid(folder, n=20):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for N, h in GRID:
        rng = np.random.default_rng(3100 + N)
        xy, conn = generate_batch(n, N, h, rng)
        dist = np.stack([euclid(v) for v in xy])
        np.savez_compressed(folder / f"N{N}_{h}.npz", xy=xy,
                            conn=conn.astype(np.uint8), dist=dist)
        print(f"wrote N{N}_{h}.npz")


if __name__ == "__main__":
    write_grid(sys.argv[1] if len(sys.argv) > 1 else "instances/grid")
