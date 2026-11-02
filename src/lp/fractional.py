"""Reading the fractionality of an LP solution (Task 2, second bullet)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

TOL = 1e-6  # "strictly between 0 and 1" means 1e-6 < value < 1 - 1e-6


def fractional_indices(x: np.ndarray, int_idx, tol: float = TOL) -> np.ndarray:
    """Indices (from int_idx) of the variables that are strictly between 0 and 1."""
    int_idx = np.asarray(int_idx, dtype=int)
    v = np.asarray(x)[int_idx]
    return int_idx[(v > tol) & (v < 1 - tol)]


def split_vms(x: np.ndarray, M: int, S: int, tol: float = TOL) -> Dict[int, List[Tuple[int, float]]]:
    """Storyline B: VMs placed on more than one server.

    y_{k,l} = x[k*S + l] (VM-major order). Returns {k: [(l, y_kl), ...]} for every VM k that has
    at least two servers with y_kl > tol, i.e. is split across several servers.
    """
    Y = np.asarray(x)[: M * S].reshape(M, S)
    out: Dict[int, List[Tuple[int, float]]] = {}
    for k in range(M):
        servers = [(l, float(Y[k, l])) for l in range(S) if Y[k, l] > tol]
        if len(servers) >= 2:
            out[k] = servers
    return out
