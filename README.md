# ConnPOMO

Code, instances and checkpoints for the paper on ConnPOMO, a POMO policy for the bus-connectivity path problem BUSCONN. Evaluation and the comparators run on a CPU. Training at the paper budget requires a GPU.

## Setup

    pip install -r requirements.txt

## Reproduce the paper

Evaluate one checkpoint on one cell of the benchmark. The preventative mask is on and the seed is the one used in the paper.

    python evaluate.py --ckpt checkpoints/N50_hard.pt --instances instances/grid/N50_hard.npz

Run the classical comparators on the same cell under the paper budget.

    python compare.py --instances instances/grid/N50_hard.npz

The real networks use the checkpoint the paper names for each set. The exact optimum comes from the Held-Karp solver.

    python evaluate.py --ckpt checkpoints/N20_medium.pt --instances instances/real/astana20_hard.npz
    python compare.py --instances instances/real/astana20_hard.npz
    python heldkarp.py instances/real/astana20_hard.npz

## Train

One model per cell. The defaults are the paper settings, 50 epochs of 100,000 instances.

    python train.py --N 50 --hardness hard

The run writes best.pt, history.json and config.json under runs/ and resumes if interrupted.

## Data

instances/grid holds thirteen cells of twenty synthetic instances with keys xy, conn and dist. `python data.py` regenerates them from the seeds. instances/real holds the Astana and Shymkent networks at 14 stops and thirty Astana instances at 20 stops, with the same keys plus latlon. Distances are Euclidean on the unit square for the grid and Haversine kilometers for the real networks.

## License

MIT.
