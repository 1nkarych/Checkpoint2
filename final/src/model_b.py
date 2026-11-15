"""Final, Block 3: the reference model of storyline B (Checkpoint 3 model) built FOR A GIVEN INSTANCE d.

    min  sum_i t_i
    (B1) sum_i x_ik = 1                               for every VM k
    (B2) sum_k p_k x_ik <= C_i                        for every server i   (CPU)
    (B3) sum_k q_k x_ik <= M_i                        for every server i   (memory)
    (B4) t_i >= a_m * (sum_k p_k x_ik) / C_i - b_m    for every server i and tangent point m
    x_ik in {0,1}, t_i >= 0; forbidden pairs (zone / GPU) are fixed to 0 through the bounds.

Variable order: x_ik at index i*M + k (server i, VM k), then t_i at S*M + i.
build_model(d) returns exactly what scipy.optimize.milp takes: c, constraints, bounds, integrality.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog
from scipy.sparse import lil_matrix

TANGENT_POINTS = (0.0, 0.4, 0.7, 0.9)


def allowed_pairs(d):
    S, M = int(d['S']), int(d['M'])
    ok = np.ones((S, M), dtype=bool)
    if d['variant'] == 2:
        for k in range(M):
            z = d['vm_required_zone'][k]
            if z != -1:
                ok[:, k] = (np.asarray(d['zone']) == z)
    elif d['variant'] == 3:
        for k in range(M):
            if d['needs_gpu'][k] == 1:
                ok[:, k] = (np.asarray(d['is_gpu_server']) == 1)
    return ok


def build_model(d):
    S, M = int(d['S']), int(d['M'])
    p = np.asarray(d['cpu_req'], float); q = np.asarray(d['mem_req'], float)
    C = np.asarray(d['cpu_cap'], float); Mm = np.asarray(d['mem_cap'], float)
    n = S * M + S
    T = len(TANGENT_POINTS)
    A = lil_matrix((M + 2 * S + T * S, n))
    lo = np.empty(M + 2 * S + T * S); hi = np.empty(M + 2 * S + T * S)
    r = 0
    for k in range(M):                                   # B1
        for i in range(S):
            A[r, i * M + k] = 1.0
        lo[r] = hi[r] = 1.0; r += 1
    for i in range(S):                                   # B2
        A[r, i * M:(i + 1) * M] = p
        lo[r], hi[r] = -np.inf, C[i]; r += 1
    for i in range(S):                                   # B3
        A[r, i * M:(i + 1) * M] = q
        lo[r], hi[r] = -np.inf, Mm[i]; r += 1
    for i in range(S):                                   # B4:  a_m * p.x / C_i - t_i <= b_m
        for um in TANGENT_POINTS:
            a = 1.0 / (1.0 - um) ** 2; b = um ** 2 / (1.0 - um) ** 2
            A[r, i * M:(i + 1) * M] = a * p / C[i]
            A[r, S * M + i] = -1.0
            lo[r], hi[r] = -np.inf, b; r += 1
    c = np.zeros(n); c[S * M:] = 1.0
    ub = np.ones(n); ub[S * M:] = np.inf
    ub[:S * M][~allowed_pairs(d).ravel()] = 0.0          # forbidden pairs
    integrality = np.zeros(n); integrality[:S * M] = 1
    return c, LinearConstraint(A.tocsr(), lo, hi), Bounds(np.zeros(n), ub), integrality


def solve_lp(d):
    """LP relaxation (0 <= x <= 1). Returns (z_LP, x as an S x M array)."""
    c, con, bounds, _ = build_model(d)
    A = con.A; eq = con.lb == con.ub
    res = linprog(c, A_ub=A[~eq], b_ub=con.ub[~eq], A_eq=A[eq], b_eq=con.ub[eq],
                  bounds=list(zip(bounds.lb, bounds.ub)), method='highs')
    S, M = int(d['S']), int(d['M'])
    return float(res.fun), res.x[:S * M].reshape(S, M)


def round_lp(d, x):
    """Our rounding rule. VMs are taken in the order of decreasing CPU requirement (ties: lower index first).
    A VM goes to the allowed server with the largest LP value x_ik that still has room in CPU and memory
    (ties: lower utilisation after the move); if no server with x_ik > 0 has room, it goes to the allowed
    server with room and the lowest CPU utilisation after the move. Returns assign (list, VM -> server)."""
    S, M = int(d['S']), int(d['M'])
    p = np.asarray(d['cpu_req'], float); q = np.asarray(d['mem_req'], float)
    C = np.asarray(d['cpu_cap'], float); Mm = np.asarray(d['mem_cap'], float)
    ok = allowed_pairs(d)
    cpu = np.zeros(S); mem = np.zeros(S); assign = [-1] * M
    for k in sorted(range(M), key=lambda k: (-p[k], k)):
        room = ok[:, k] & (cpu + p[k] <= C + 1e-9) & (mem + q[k] <= Mm + 1e-9)
        if not room.any():
            raise RuntimeError(f'rounding failed: no server with room for VM {k}')
        util = (cpu + p[k]) / C
        cand = np.nonzero(room & (x[:, k] > 1e-9))[0]
        if len(cand):
            i = min(cand, key=lambda i: (-round(x[i, k], 9), util[i], i))
        else:
            cand = np.nonzero(room)[0]
            i = min(cand, key=lambda i: (util[i], i))
        assign[k] = int(i); cpu[i] += p[k]; mem[i] += q[k]
    return assign


def x_to_assign(d, xvec):
    S, M = int(d['S']), int(d['M'])
    return [int(v) for v in np.argmax(np.asarray(xvec)[:S * M].reshape(S, M), axis=0)]
