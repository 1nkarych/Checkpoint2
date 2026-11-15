"""Sparse, instance-parameterized adaptation of T10 CP3 model_builder.py.

Original: src/model/model_builder.py used cp34_instance(10,'B',2), dense matrices,
and hard-coded dimensions. This adaptation retains EXACTLY its mathematical
formulation (B1 assignment, B2 CPU, B3 memory, B4 tangent latency, zone bounds)
but accepts the final-kit scaled instance d and uses scipy.sparse for M=400.

Column order: y[k,l], k-major, then t[l]. Objective = sum_l t[l].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint
from checkpoint4_generator_PUBLIC import TANGENT_POINTS


@dataclass
class Model:
    d: dict
    c: np.ndarray
    Aeq: sp.csr_matrix
    beq: np.ndarray
    Aub: sp.csr_matrix
    bub: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray

    @property
    def S(self): return int(self.d['S'])
    @property
    def M(self): return int(self.d['M'])

    def linprog_kwargs(self):
        return dict(c=self.c, A_ub=self.Aub, b_ub=self.bub,
                    A_eq=self.Aeq, b_eq=self.beq,
                    bounds=list(zip(self.lb, self.ub)))

    def milp_kwargs(self):
        A = sp.vstack((self.Aeq, self.Aub), format='csr')
        row_lo = np.r_[self.beq, np.full(len(self.bub), -np.inf)]
        row_hi = np.r_[self.beq, self.bub]
        return dict(c=self.c, integrality=self.integrality,
                    bounds=Bounds(self.lb, self.ub),
                    constraints=LinearConstraint(A, row_lo, row_hi))


def build_model(d: dict) -> Model:
    """Build the exact CP3 MILP/LP matrix for any scaled B2 instance."""
    if (d.get('story'), d.get('variant')) != ('B', 2):
        raise ValueError('Team T10 uses Storyline B, Variant 2 only')
    M, S = int(d['M']), int(d['S'])
    n_y, n = M * S, M * S + S
    c = np.zeros(n, dtype=float); c[n_y:] = 1.0
    lb = np.zeros(n, dtype=float); ub = np.full(n, np.inf)
    ub[:n_y] = 1.0
    integrality = np.zeros(n, dtype=np.int8); integrality[:n_y] = 1
    # CP3 forbids wrong-zone assignments by setting y[k,l] = 0.
    for k in range(M):
        z = int(d['vm_required_zone'][k])
        if z >= 0:
            for l in range(S):
                if int(d['zone'][l]) != z:
                    ub[k*S+l] = 0.0
    # B1: exactly one server per VM.
    Aeq = sp.kron(sp.eye(M, format='csr'),
                  sp.csr_matrix(np.ones((1, S))), format='csr')
    Aeq = sp.hstack((Aeq, sp.csr_matrix((M, S))), format='csr')
    beq = np.ones(M, dtype=float)
    # B2 CPU; B3 RAM; B4 each tangent lower envelope linearization.
    rows, cols, vals = [], [], []
    bub = []
    r = 0
    for resource, capacity in (('cpu_req', 'cpu_cap'), ('mem_req', 'mem_cap')):
        for l in range(S):
            for k in range(M):
                rows.append(r); cols.append(k*S+l); vals.append(float(d[resource][k]))
            bub.append(float(d[capacity][l])); r += 1
    for l in range(S):
        for a in TANGENT_POINTS:
            slope = 1.0/(1.0-float(a))**2
            for k in range(M):
                rows.append(r); cols.append(k*S+l)
                vals.append(slope * float(d['cpu_req'][k])/float(d['cpu_cap'][l]))
            rows.append(r); cols.append(n_y+l); vals.append(-1.0)
            # t >= slope*u - a^2/(1-a)^2
            bub.append(slope*float(a) - float(a)/(1.0-float(a)))
            r += 1
    Aub = sp.coo_matrix((np.asarray(vals), (rows, cols)), shape=(r, n)).tocsr()
    model = Model(d, c, Aeq, beq, Aub, np.asarray(bub), lb, ub, integrality)
    assert model.Aeq.shape == (M, n)
    assert model.Aub.shape == (S*(2+len(TANGENT_POINTS)), n)
    assert np.allclose(np.asarray(Aeq.sum(axis=1)).ravel(), S)
    assert np.allclose(np.asarray(Aeq[:, :n_y].sum(axis=1)).ravel(), S)
    return model
