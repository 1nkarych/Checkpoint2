"""One reproducible, capacity- and zone-feasible LP-guided starting assignment.

Rounding rule (describe it in the report): Solve the CP3 LP relaxation.
Process VMs with larger LP confidence first, preferring required-zone VMs.
For each VM, select a feasible allowed server maximizing fractional y_kl;
within ties choose lower incremental convex piecewise-linear latency.
Retry with alternate deterministic tie-breaking if necessary, but NEVER
silently replace an LP start with an unrelated SA/GA own start.
"""
from __future__ import annotations
import numpy as np
from checkpoint4_generator_PUBLIC import evaluate_B, latency_B


def round_lp(d: dict, lp_x: np.ndarray, *, attempts: int = 40) -> tuple[list[int], dict]:
    M, S = int(d['M']), int(d['S'])
    y = np.asarray(lp_x[:M*S], dtype=float).reshape(M, S)
    if np.any(~np.isfinite(y)):
        raise ValueError('LP solution contains non-finite values')
    allowed = []
    for k in range(M):
        z = int(d['vm_required_zone'][k])
        opts = [l for l in range(S) if z < 0 or int(d['zone'][l]) == z]
        if not opts: raise ValueError(f'VM {k} has no allowed server')
        allowed.append(opts)
    cpu_req = np.asarray(d['cpu_req'], dtype=float)
    mem_req = np.asarray(d['mem_req'], dtype=float)
    cpu_cap = np.asarray(d['cpu_cap'], dtype=float)
    mem_cap = np.asarray(d['mem_cap'], dtype=float)
    margins = np.array([sorted((y[k, l] for l in allowed[k]), reverse=True)[:2]
                        for k in range(M)], dtype=object)
    conf = np.asarray([float(q[0] - q[1]) if len(q)>1 else 1.0 for q in margins])
    # Best-confidence rows first, zone-restricted earlier in otherwise similar rows.
    base_order = sorted(range(M), key=lambda k: (-conf[k], len(allowed[k]),
                                                -cpu_req[k]/cpu_cap.mean()-mem_req[k]/mem_cap.mean(), k))
    for trial in range(attempts):
        rng = np.random.default_rng(20261115 + trial)
        order = list(base_order)
        if trial:
            # Still LP-guided: randomized tie-break around the original confidence ordering.
            order = sorted(order, key=lambda k: (-conf[k]+0.1*rng.random(), len(allowed[k]), k))
        cpu = np.zeros(S); mem = np.zeros(S)
        assign = [-1]*M
        failed = False
        for k in order:
            feasible = [l for l in allowed[k] if
                        cpu[l]+cpu_req[k] <= cpu_cap[l]+1e-9 and
                        mem[l]+mem_req[k] <= mem_cap[l]+1e-9]
            if not feasible:
                failed = True; break
            # Prefer LP mass first, marginal-latency second (explicit deterministic rule).
            def key(l):
                old = cpu[l]/cpu_cap[l]
                new = (cpu[l]+cpu_req[k])/cpu_cap[l]
                delta = latency_B(new)-latency_B(old)
                return (-round(float(y[k, l]), 9), delta,
                        (mem[l]+mem_req[k])/mem_cap[l], l)
            chosen = min(feasible, key=key)
            assign[k] = int(chosen)
            cpu[chosen] += cpu_req[k]; mem[chosen] += mem_req[k]
        if failed: continue
        cost, problems = evaluate_B(d, assign)
        if not problems:
            return assign, {'rounding_rule': 'confidence-priority LP-guided feasible greedy',
                            'trial': trial, 'feasible': True, 'cost': float(cost)}
    raise RuntimeError(f'LP-guided rounding failed after {attempts} attempts; '
                       'report this honestly rather than silently using a different start')
