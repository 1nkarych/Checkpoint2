"""GA-P part (storyline B, variant B2: cloud with zones) -- problem-specific functions of the GA.

Authored by the GA-P role. Agreed with GA-A (ga.py) -- all functions use numpy arrays and a numpy Generator `rng`:

    prepare(d)                      -> ctx   (precomputed data; read-only, call once per solve)
    random_individual(ctx, rng)     -> int array of length M (server of every VM), zone-valid, NOT repaired
    repair(ctx, ind, rng)           -> int array, capacity- and zone-feasible (reads data only: 0 evaluations)
    evaluate(ctx, ind, rho=None)    -> (fitness, cost, feasible)   [one evaluation for the budget counter, counted by GA-A]
    crossover(ctx, p1, p2, rng)     -> one child (uniform crossover)
    mutate(ctx, ind, rng)           -> new individual, 1 or 2 genes reassigned to an allowed server

Configuration "GA, repair": GA-A calls repair() on every offspring BEFORE evaluate(); fitness == cost.
For the experiment "repair off" GA-A skips repair() and calls evaluate(ctx, ind, rho=RHO): fitness = cost + rho * violation.
"""
import numpy as np

TANGENT_POINTS = (0.0, 0.4, 0.7, 0.9)           # same as the checker
_A = np.array([1.0 / (1.0 - um) ** 2 for um in TANGENT_POINTS])
_B = np.array([um ** 2 / (1.0 - um) ** 2 for um in TANGENT_POINTS])


def prepare(d):
    S, M = d['S'], d['M']
    allowed = []
    for k in range(M):
        if d['variant'] == 2 and d['vm_required_zone'][k] != -1:
            allowed.append(np.where(d['zone'] == d['vm_required_zone'][k])[0])
        else:
            allowed.append(np.arange(S))
    return dict(S=S, M=M, cpu_cap=np.asarray(d['cpu_cap'], float), mem_cap=np.asarray(d['mem_cap'], float),
                cpu_req=np.asarray(d['cpu_req'], float), mem_req=np.asarray(d['mem_req'], float),
                allowed=allowed, d=d)


def loads(ctx, ind):
    cpu = np.bincount(ind, weights=ctx['cpu_req'], minlength=ctx['S'])
    mem = np.bincount(ind, weights=ctx['mem_req'], minlength=ctx['S'])
    return cpu, mem


def cost_of(ctx, cpu):
    u = np.minimum(cpu / ctx['cpu_cap'], 1.0)
    return float(np.max(u[:, None] * _A[None, :] - _B[None, :], axis=1).sum())   # same formula as latency_B


def violation(ctx, cpu, mem):
    """Sum of the relative overloads (CPU and memory); 0 for a feasible solution."""
    return float(np.maximum(0.0, cpu - ctx['cpu_cap']).__truediv__(ctx['cpu_cap']).sum()
                 + np.maximum(0.0, mem - ctx['mem_cap']).__truediv__(ctx['mem_cap']).sum())


def zone_ok(ctx, ind):
    return all(ind[k] in ctx['allowed'][k] for k in range(ctx['M']))


def evaluate(ctx, ind, rho=None):
    cpu, mem = loads(ctx, ind)
    cost = cost_of(ctx, cpu)
    v = violation(ctx, cpu, mem)
    feasible = (v <= 1e-12) and zone_ok(ctx, ind)
    fitness = cost if rho is None else cost + rho * (v + (0.0 if zone_ok(ctx, ind) else 1.0))
    return fitness, cost, feasible


def random_individual(ctx, rng):
    return np.array([rng.choice(ctx['allowed'][k]) for k in range(ctx['M'])], dtype=int)


def crossover(ctx, p1, p2, rng):
    mask = rng.random(ctx['M']) < 0.5
    return np.where(mask, p1, p2).astype(int)       # both parents zone-valid -> child zone-valid


def mutate(ctx, ind, rng):
    child = ind.copy()
    for k in rng.choice(ctx['M'], size=int(rng.integers(1, 3)), replace=False):
        opts = ctx['allowed'][k]
        opts = opts[opts != child[k]] if len(opts) > 1 else opts
        child[k] = rng.choice(opts)
    return child


# ------------------------------------------------------------------ repair
def _fits(ctx, cpu, mem, k, l):
    return cpu[l] + ctx['cpu_req'][k] <= ctx['cpu_cap'][l] and mem[l] + ctx['mem_req'][k] <= ctx['mem_cap'][l]


def _rebuild(ctx, rng):
    """Fallback: first-fit-decreasing in zone-restricted order; best server = lowest CPU utilisation after placing."""
    cpu = np.zeros(ctx['S']); mem = np.zeros(ctx['S'])
    ind = np.zeros(ctx['M'], dtype=int)
    order = sorted(range(ctx['M']), key=lambda k: (len(ctx['allowed'][k]), -(ctx['mem_req'][k] + ctx['cpu_req'][k])))
    for k in order:
        best, bu = None, 1e18
        for l in ctx['allowed'][k]:
            if _fits(ctx, cpu, mem, k, l):
                u = (cpu[l] + ctx['cpu_req'][k]) / ctx['cpu_cap'][l]
                if u < bu: best, bu = l, u
        if best is None:                                # no room anywhere: least overloaded allowed server
            best = min(ctx['allowed'][k], key=lambda l: max((cpu[l] + ctx['cpu_req'][k]) / ctx['cpu_cap'][l],
                                                             (mem[l] + ctx['mem_req'][k]) / ctx['mem_cap'][l]))
        ind[k] = best; cpu[best] += ctx['cpu_req'][k]; mem[best] += ctx['mem_req'][k]
    return ind


def repair(ctx, ind, rng):
    """Repair rule (reads data only, never calls the cost function):
    1. every VM outside its required zone goes to a random server of its zone;
    2. while a server is overloaded in CPU or memory: take the most overloaded server (largest relative overload);
       among its VMs choose the one with the SMALLEST size that removes the excess in the violated resource(s)
       (if none is big enough, the largest one), and move it to the allowed server with enough room in BOTH
       resources and the LOWEST CPU utilisation after the move (convex latency -> balanced load is cheap);
       VMs that fit nowhere are skipped;
    3. if no VM can be moved (or after 5*M steps) the fallback rebuilds the whole assignment greedily (FFD).
    The result is feasible whenever the instance leaves enough room; otherwise the checker reports it honestly."""
    ind = np.asarray(ind, dtype=int).copy()
    for k in range(ctx['M']):
        if ind[k] not in ctx['allowed'][k]:
            ind[k] = rng.choice(ctx['allowed'][k])
    cpu, mem = loads(ctx, ind)
    for _ in range(5 * ctx['M']):
        ex_c = np.maximum(0.0, cpu - ctx['cpu_cap']); ex_m = np.maximum(0.0, mem - ctx['mem_cap'])
        rel = ex_c / ctx['cpu_cap'] + ex_m / ctx['mem_cap']
        if rel.max() <= 1e-12:
            return ind
        l = int(np.argmax(rel))
        vms = np.where(ind == l)[0]
        enough = [k for k in vms if ctx['cpu_req'][k] >= ex_c[l] and ctx['mem_req'][k] >= ex_m[l]]
        cands = sorted(enough, key=lambda k: ctx['cpu_req'][k] / ctx['cpu_cap'][l] + ctx['mem_req'][k] / ctx['mem_cap'][l]) \
            + sorted([k for k in vms if k not in enough],
                     key=lambda k: -(ctx['cpu_req'][k] / ctx['cpu_cap'][l] + ctx['mem_req'][k] / ctx['mem_cap'][l]))
        moved = False
        for k in cands:
            best, bu = None, 1e18
            for t in ctx['allowed'][k]:
                if t != l and _fits(ctx, cpu, mem, k, t):
                    u = (cpu[t] + ctx['cpu_req'][k]) / ctx['cpu_cap'][t]
                    if u < bu: best, bu = t, u
            if best is not None:
                cpu[l] -= ctx['cpu_req'][k]; mem[l] -= ctx['mem_req'][k]
                cpu[best] += ctx['cpu_req'][k]; mem[best] += ctx['mem_req'][k]
                ind[k] = best; moved = True
                break
        if not moved:
            break
    cpu, mem = loads(ctx, ind)
    if violation(ctx, cpu, mem) > 1e-12:
        ind = _rebuild(ctx, rng)
    return ind
