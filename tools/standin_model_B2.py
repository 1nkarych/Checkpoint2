"""STAND-IN for M1's src/model (NOT part of M2's authored code).
Builds the reference model B2 in matrix form only so M2's LP code can be run and verified.
Replace by M1's builder when it is merged. Variable order: y_{k,l} = k*S+l, then t_l = M*S+l."""
import numpy as np
from checkpoint3_generator_PUBLIC import cp34_instance, TANGENT_POINTS
from src.lp import LPData


def build_B2(seed=10):
    d = cp34_instance(seed, 'B', 2)
    S, M = d['S'], d['M']
    cpu, mem, C, R = d['cpu_req'], d['mem_req'], d['cpu_cap'], d['mem_cap']
    n = M * S + S
    c = np.zeros(n); c[M * S:] = 1.0
    A_eq = np.zeros((M, n)); b_eq = np.ones(M)
    for k in range(M):
        A_eq[k, k * S:(k + 1) * S] = 1
    rows, rhs = [], []
    for l in range(S):                       # B2
        r = np.zeros(n); r[[k * S + l for k in range(M)]] = cpu; rows.append(r); rhs.append(C[l])
    for l in range(S):                       # B3
        r = np.zeros(n); r[[k * S + l for k in range(M)]] = mem; rows.append(r); rhs.append(R[l])
    for l in range(S):                       # B4
        for u in TANGENT_POINTS:
            a = 1 / (1 - u) ** 2; b = -u ** 2 / (1 - u) ** 2
            r = np.zeros(n)
            r[[k * S + l for k in range(M)]] = a * cpu / C[l]
            r[M * S + l] = -1.0
            rows.append(r); rhs.append(-b)
    bounds = []
    for k in range(M):
        for l in range(S):
            forb = d['vm_required_zone'][k] != -1 and d['zone'][l] != d['vm_required_zone'][k]
            bounds.append((0.0, 0.0 if forb else 1.0))
    bounds += [(0.0, None)] * S
    return LPData(c, np.array(rows), np.array(rhs), A_eq, b_eq, bounds), d
