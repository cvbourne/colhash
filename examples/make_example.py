"""Build the files used in the README worked example.

A made-up pipeline totals each user's purchases. Summing floats in a different order gives
slightly different last bits, as it does when a job runs in parallel, so two runs of the
*same* pipeline disagree a little. Three kinds of candidate follow:

  baseline_1, baseline_2   the unchanged pipeline, run twice (different summation order)
  refactor                 an equivalent rewrite, again with its own summation order
  bug                      a rewrite that silently drops purchases under $0.05

Run:  python examples/make_example.py examples/data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_USERS, N_EVENTS = 2_000, 400_000


def events(seed=0):
    rng = np.random.default_rng(seed)
    user = rng.integers(0, N_USERS, N_EVENTS)
    amount = np.round(rng.lognormal(mean=2.0, sigma=1.5, size=N_EVENTS), 4)
    return user, amount


def pipeline(user, amount, order_seed, min_amount=0.0):
    keep = amount >= min_amount
    user, amount = user[keep], amount[keep]
    order = np.random.default_rng(order_seed).permutation(len(user))  # stands in for a parallel schedule
    total = np.zeros(N_USERS)
    np.add.at(total, user[order], amount[order])  # sequential, so summation order matters
    n = np.bincount(user, minlength=N_USERS)
    return pa.table({
        "user_id": pa.array(np.arange(N_USERS)),
        "total_spend": pa.array(total),
        "n_purchases": pa.array(n.astype(np.int64)),
        "tier": pa.array(np.where(n > 220, "high", "standard")),
    })


def main(out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    user, amount = events()
    runs = {
        "baseline_1": pipeline(user, amount, order_seed=1),
        "baseline_2": pipeline(user, amount, order_seed=2),
        "refactor": pipeline(user, amount, order_seed=3),
        "bug": pipeline(user, amount, order_seed=4, min_amount=0.05),
    }
    for name, table in runs.items():
        pq.write_table(table, out_dir / f"{name}.parquet")
    print(f"wrote {', '.join(runs)} to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/data")
