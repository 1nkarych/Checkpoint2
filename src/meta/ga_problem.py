"""GA-P part: representation and problem-specific operators for storyline B (cloud), repair configuration.

    random_individual(d, rng)            -> feasible individual (random assignment + repair)
    evaluate(d, ind)                     -> (fitness, cost, feasible); fitness = cost + RHO*violation (lower is better)
    crossover(d, p1, p2, rng)            -> one child (uniform crossover)
    mutate(d, ind, rng, n)               -> copy of ind with n components moved to another allowed server
    repair(d, ind, rng)                  -> feasible individual (reads loads and capacities only: free of charge)
    violation(d, ind)                    -> total violation (0 for feasible)
"""
import numpy as np

RHO = 100.0   # safety net only: the repair delivers feasible individuals, so fitness == cost in normal operation
EPS = 1e-9
TANGENTS = (0.0, 0.4, 0.7, 0.9)   # same as the generator (fixed for everyone)


def _ctx(d):
    if d.get('_is_ctx', False):
        return d
    c = d.get('_ga_ctx')
    if c is not None:
        return c
    S, M = int(d['S']), int(d['M'])
    allowed = np.ones((M, S), dtype=bool)
    if d['variant'] == 2:
        for k in range(M):
            z = d['vm_required_zone'][k]
            if z != -1:
                allowed[k] = (np.asarray(d['zone']) == z)
    elif d['variant'] == 3:
        for k in range(M):
            if d['needs_gpu'][k] == 1:
                allowed[k] = (np.asarray(d['is_gpu_server']) == 1)
    c = dict(
        S=S, M=M,
        cpu_req=np.asarray(d['cpu_req'], dtype=float), mem_req=np.asarray(d['mem_req'], dtype=float),
        cpu_cap=np.asarray(d['cpu_cap'], dtype=float), mem_cap=np.asarray(d['mem_cap'], dtype=float),
        allowed=allowed, allowed_list=[np.nonzero(allowed[k])[0] for k in range(M)],
        a=np.array([1.0 / (1.0 - um) ** 2 for um in TANGENTS]),
        b=np.array([um ** 2 / (1.0 - um) ** 2 for um in TANGENTS]),
        ar=np.arange(M),
    )

    d['_ga_ctx'] = c
    return c


def prepare(d):
    """Build and return the context expected by the GA-P tests."""
    c = _ctx(d)
    c['_is_ctx'] = True
    return c


def _loads(c, ind):
    cpu = np.bincount(ind, weights=c['cpu_req'], minlength=c['S'])
    mem = np.bincount(ind, weights=c['mem_req'], minlength=c['S'])
    return cpu, mem


def loads(ctx, ind):
    """Return CPU and memory loads for each server."""
    return _loads(_ctx(ctx), np.asarray(ind, dtype=np.int64))


def zone_ok(ctx, ind):
    """Check whether every VM is assigned to an allowed server."""
    c = _ctx(ctx)
    x = np.asarray(ind, dtype=np.int64)
    return bool(c['allowed'][c['ar'], x].all())


def violation(d, ind):
    c = _ctx(d)
    cpu, mem = _loads(c, ind)
    v = np.maximum(0.0, cpu - c['cpu_cap'] - EPS) / c['cpu_cap']
    v = v.sum() + (np.maximum(0.0, mem - c['mem_cap'] - EPS) / c['mem_cap']).sum()
    v += float((~c['allowed'][c['ar'], ind]).sum())   # placements that break the zone / GPU condition
    return float(v)


def evaluate(d, ind, rho=None):
    c = _ctx(d)
    cpu, mem = _loads(c, ind)
    u = np.minimum(cpu / c['cpu_cap'], 1.0)
    cost = float(np.max(u[:, None] * c['a'][None, :] - c['b'][None, :], axis=1).sum())
    viol = violation(d, ind)
    fitness = cost if rho is None else cost + float(rho) * viol
    return fitness, cost, viol == 0.0


def crossover(d, p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    return np.where(mask, p1, p2)


def mutate(d, ind, rng, n=None):
    c = _ctx(d)
    if n is None:
        n = int(rng.integers(1, 3))
    x = ind.copy()
    for k in rng.choice(c['M'], size=min(n, c['M']), replace=False):
        opts = c['allowed_list'][k]
        opts = opts[opts != x[k]]
        if len(opts):
            x[k] = opts[rng.integers(len(opts))]
    return x


def repair(d, ind, rng):
    """1) VMs on a forbidden server (zone / GPU) go to a random allowed server.
    2) while some server exceeds CPU or memory: take an overloaded server; among its VMs pick the smallest one whose
       removal cures the overload (otherwise the largest one) that has a target with room; the target is the allowed
       server with room and the lowest CPU utilisation after the move (balancing = low latency).
    Only data are read (loads, capacities): no call of the cost function."""
    c = _ctx(d)
    x = ind.astype(np.int64, copy=True)
    bad = np.nonzero(~c['allowed'][c['ar'], x])[0]
    for k in bad:
        x[k] = c['allowed_list'][k][rng.integers(len(c['allowed_list'][k]))]
    cpu, mem = _loads(c, x)
    cc, mc = c['cpu_cap'], c['mem_cap']
    for _ in range(4 * c['M']):
        over = np.nonzero((cpu > cc + EPS) | (mem > mc + EPS))[0]
        if len(over) == 0:
            return x
        l = int(over[rng.integers(len(over))])
        ocpu, omem = max(0.0, cpu[l] - cc[l]), max(0.0, mem[l] - mc[l])
        members = np.nonzero(x == l)[0]
        best = None   # (key, k, target)
        for k in members[rng.permutation(len(members))]:
            ok = c['allowed'][k] & (cpu + c['cpu_req'][k] <= cc + EPS) & (mem + c['mem_req'][k] <= mc + EPS)
            ok[l] = False
            if not ok.any():
                continue
            size = c['cpu_req'][k] / cc[l] + c['mem_req'][k] / mc[l]
            cures = (c['cpu_req'][k] >= ocpu - EPS) and (c['mem_req'][k] >= omem - EPS)
            key = (0, size) if cures else (1, -size)
            if best is None or key < best[0]:
                util = np.where(ok, (cpu + c['cpu_req'][k]) / cc, np.inf)
                best = (key, int(k), int(np.argmin(util)))
        if best is None:
            return x   # cannot repair (should not happen at moderate fill); the remaining violation will be reported
        _, k, t = best
        cpu[l] -= c['cpu_req'][k]; mem[l] -= c['mem_req'][k]
        cpu[t] += c['cpu_req'][k]; mem[t] += c['mem_req'][k]
        x[k] = t
    return x


def _rebuild(ctx, ind, rng):
    """Fallback rebuild helper; uses the same data-only repair procedure."""
    return repair(ctx, ind, rng)


_repair_fn = repair   # у random_individual есть параметр repair, он скрывает функцию, поэтому нужен псевдоним


def random_individual(d, rng, repair=True):
    c = _ctx(d)
    if not repair:
        return np.array([opts[rng.integers(len(opts))] for opts in c['allowed_list']], dtype=np.int64)
    for _ in range(50):
        x = np.array([opts[rng.integers(len(opts))] for opts in c['allowed_list']], dtype=np.int64)
        x = _repair_fn(d, x, rng)
        if violation(d, x) == 0.0:
            return x
    return x