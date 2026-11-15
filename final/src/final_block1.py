"""Final report, Block 1 (team T10, instance cp34_instance(10, 'B', 2)): Checkpoint 1 objective vs reference model.

ONE script for the whole Block 1 computation (it replaces the former cp1_eval.py + final_block1.py pair; the old files are kept
in legacy/ for the history only).

What it does
  default            evaluates the three STORED assignments of results/final_block1_solutions.json
                     (reference optimum returned by HiGHS, best GA run = seed 4, best SA run = seed 3)
                     under (a) the reference cost (checker evaluate_B) and (b) the Checkpoint 1 objective
                         f(x) = sum_j (U_j + U_j^2),  U_j = max(U_cpu_j, U_mem_j),
                     checks the CP1 constraints and writes results/final_block1_eval.csv and
                     results/final_block1_decomposition.csv. It then compares the three values with the control values
                     quoted in the report and stops with exit code 1 if they differ by more than 5e-7 (a regression check;
                     nothing is fitted to them).
  --resolve-reference  re-solves the reference model with HiGHS (scipy.optimize.milp), reports z_ref, z_LP and whether the
                     solver returns the stored assignment (the optimum is not unique, so another HiGHS version may return
                     another one). Writes results/final_block1_reference_resolve.json.
  --optimal-set      (M1) fixes the reference cost to z_ref + 1e-6 and minimises / maximises the CP1 objective over all
                     assignments that satisfy it. Both are MILPs with a piecewise-linear model of f; the script reports the
                     MILP bound and the TRUE CP1 value of the incumbent, so the result is an interval, not a guess.
                     Writes results/final_block1_optimal_set.json.
  --seeds            needs the team code (src/meta ga, sa, interface; final_block2.read, sa_params_from_csv; the run CSVs).
                     Re-runs seeds 0-9 of GA and SA and writes results/final_block1_seeds.csv with the CP1 objective of
                     every final solution. NOT available in the working copy that was used for the revision of this report.

Usage (repository root):
    python final/src/final_block1.py
    python final/src/final_block1.py --resolve-reference --optimal-set
    python final/src/final_block1.py --seeds                  # only with the team code present
"""
import argparse, csv, json, os, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint4_generator_PUBLIC as gen

R = os.path.join(ROOT, 'results')
INST = dict(seed=10, story='B', variant=2)
BUDGET = 100000
REF_KEY = 'Reference optimum (exact)'
# control values quoted in the report (Table of Block 1); used ONLY as a regression check
CONTROL = {REF_KEY: 11.294574, 'GA best seed': 10.688779, 'SA best seed': 11.426841}


def instance():
    return gen.cp34_instance(INST['seed'], INST['story'], INST['variant'])


# ------------------------------------------------------------------------------------------- Checkpoint 1 objective
def cp1_parts(d, assign):
    """Checkpoint 1 model: f(x) = sum_j (U_j + U_j^2), U_j = max(U_cpu_j, U_mem_j).
    Returns value, U, CPU and memory utilisation per server, and violations of the CP1 constraints."""
    S, M = d['S'], d['M']
    a = np.asarray(assign, dtype=int)
    assert a.shape == (M,) and a.min() >= 0 and a.max() < S, 'assignment must be 0-based server indices, length M'
    cpu = np.bincount(a, weights=np.asarray(d['cpu_req'], float), minlength=S)
    mem = np.bincount(a, weights=np.asarray(d['mem_req'], float), minlength=S)
    ucpu = cpu / np.asarray(d['cpu_cap'], float); umem = mem / np.asarray(d['mem_cap'], float)
    U = np.maximum(ucpu, umem)
    zone_viol = sum(1 for k in range(M) if d['vm_required_zone'][k] != -1 and d['zone'][a[k]] != d['vm_required_zone'][k])
    viol = dict(cpu=int((ucpu > 1 + 1e-9).sum()), mem=int((umem > 1 + 1e-9).sum()), zone=int(zone_viol))
    return float(np.sum(U + U ** 2)), U, ucpu, umem, viol


def cp1_objective(d, assign):
    f, U, ucpu, umem, viol = cp1_parts(d, assign)
    assert not any(viol.values()), f'CP1 constraints violated: {viol}'   # CP1 and reference feasible sets coincide
    return f


def cp1_cpu_only(d, assign):
    """Decomposition helper (NOT the CP1 objective): sum_j (u_j + u_j^2) with the CPU utilisation only."""
    f, U, ucpu, umem, viol = cp1_parts(d, assign)
    return float(np.sum(ucpu + ucpu ** 2))


# ------------------------------------------------------------------------------------------- reference model (MILP)
class Layout:
    """Variable layout: y[k,l] (M*S), t[l] (S), then extra blocks appended by the caller."""
    def __init__(self, d):
        self.S, self.M = d['S'], d['M']; self.ny = self.M * self.S; self.n = self.ny + self.S
    def y(self, k, l): return k * self.S + l
    def t(self, l): return self.ny + l
    def add(self, count):
        i = self.n; self.n += count; return i


def reference_rows(d, L, relax=False):
    """Rows of the reference model of Checkpoint 3 (storyline B, variant 2); returns (list of (row dict, lo, hi)), bounds."""
    S, M = d['S'], d['M']
    rows = []
    for k in range(M):                                                 # each VM on exactly one server
        rows.append(({L.y(k, l): 1.0 for l in range(S)}, 1.0, 1.0))
    for l in range(S):                                                 # CPU and memory capacity
        rows.append(({L.y(k, l): float(d['cpu_req'][k]) for k in range(M)}, -np.inf, float(d['cpu_cap'][l])))
        rows.append(({L.y(k, l): float(d['mem_req'][k]) for k in range(M)}, -np.inf, float(d['mem_cap'][l])))
    for l in range(S):                                                 # latency tangents: t_l >= slope*u_l + intercept
        for um in gen.TANGENT_POINTS:
            slope = 1.0 / (1.0 - um) ** 2; icpt = -um ** 2 / (1.0 - um) ** 2
            r = {L.y(k, l): slope * float(d['cpu_req'][k]) / float(d['cpu_cap'][l]) for k in range(M)}
            r[L.t(l)] = -1.0
            rows.append((r, -np.inf, -icpt))
    return rows


def base_bounds(d, L):
    lb = np.zeros(L.n); ub = np.full(L.n, np.inf); ub[:L.ny] = 1.0
    nforb = 0
    for k in range(d['M']):                                            # zone restriction: forbidden pairs fixed to 0
        zr = d['vm_required_zone'][k]
        if zr != -1:
            for l in range(d['S']):
                if d['zone'][l] != zr: ub[L.y(k, l)] = 0.0; nforb += 1
    return lb, ub, nforb


def run_milp(L, rows, c, lb, ub, integrality, time_limit, label=''):
    A = lil_matrix((len(rows), L.n)); lo = np.empty(len(rows)); hi = np.empty(len(rows))
    for i, (r, a, b) in enumerate(rows):
        for j, v in r.items(): A[i, j] = v
        lo[i], hi[i] = a, b
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), bounds=Bounds(lb, ub), integrality=integrality,
               options=dict(time_limit=time_limit, disp=False, mip_rel_gap=1e-9))
    return res, time.time() - t0


def solve_reference(d, relax=False, time_limit=120):
    L = Layout(d); rows = reference_rows(d, L); lb, ub, nforb = base_bounds(d, L)
    c = np.zeros(L.n); c[L.ny:L.ny + L.S] = 1.0
    integ = np.zeros(L.n); 
    if not relax: integ[:L.ny] = 1
    res, sec = run_milp(L, rows, c, lb, ub, integ, time_limit)
    out = dict(status=int(res.status), message=res.message, value=None if res.fun is None else float(res.fun), seconds=sec,
               forbidden_pairs=nforb, rows=len(rows), variables=L.n)
    if not relax and res.x is not None:
        y = res.x[:L.ny].reshape(L.M, L.S)
        out['assign'] = [int(v) for v in y.argmax(axis=1)]
        out['mip_gap'] = getattr(res, 'mip_gap', None)
    return out


# ------------------------------------------------------------------------------------------- M1: CP1 over the optimal set
def optimal_set_range(d, zref, sense, time_limit, K=100, tol=1e-6):
    """min or max of the CP1 objective over {assignments with reference cost <= zref + tol}.
    sense = 'min': U_j >= max(ucpu, umem); w_j >= tangent_k(U_j) of f(U) = U + U^2 (K tangents, a LOWER model of f)
                   -> MILP optimum is a valid lower bound of the true minimum.
    sense = 'max': U_j <= max(ucpu, umem) via a binary b_j; f(U_j) as the linear interpolation of f on K equal segments
                   (incremental formulation with binaries, an UPPER model of the convex f)
                   -> MILP optimum is a valid upper bound of the true maximum.
    The true CP1 value of the incumbent is returned as well (a value that IS attained)."""
    S, M = d['S'], d['M']
    L = Layout(d); rows = reference_rows(d, L); lb, ub, nforb = base_bounds(d, L)
    rows.append(({L.t(l): 1.0 for l in range(S)}, -np.inf, zref + tol))          # reference cost at the optimum
    cpu_cap = np.asarray(d['cpu_cap'], float); mem_cap = np.asarray(d['mem_cap'], float)
    iU = L.add(S); iF = L.add(S)
    lb = np.concatenate([lb, np.zeros(2 * S)]); ub = np.concatenate([ub, np.full(S, 1.0), np.full(S, np.inf)])
    integ = np.zeros(L.n + 0); 
    def ucpu_row(l, coef):                       # coef * U_cpu_l as dict
        return {L.y(k, l): coef * float(d['cpu_req'][k]) / cpu_cap[l] for k in range(M)}
    def umem_row(l, coef):
        return {L.y(k, l): coef * float(d['mem_req'][k]) / mem_cap[l] for k in range(M)}
    c = np.zeros(L.n)
    if sense == 'min':
        for l in range(S):
            for u_ in (ucpu_row, umem_row):                      # U_l - U_cpu/U_mem >= 0
                r = u_(l, -1.0); r[iU + l] = 1.0; rows.append((r, 0.0, np.inf))
            for kk in range(K + 1):                              # F_l >= tangent of U+U^2 at x: (1+2x)U - x^2
                x = kk / K
                rows.append(({iF + l: 1.0, iU + l: -(1 + 2 * x)}, -x * x, np.inf))
        c[iF:iF + S] = 1.0
        integ_full = np.zeros(L.n); integ_full[:L.ny] = 1
        res, sec = run_milp(L, rows, c, lb, ub, integ_full, time_limit)
        sign = 1.0
    else:
        ib = L.add(S); idel = L.add(S * K); iz = L.add(S * (K - 1))
        lb = np.concatenate([lb, np.zeros(S + S * K + S * (K - 1))])
        ub = np.concatenate([ub, np.ones(S + S * K + S * (K - 1))])
        c = np.zeros(L.n)
        xs = np.arange(K + 1) / K; fs = xs + xs ** 2; slope = np.diff(fs) * K        # f increments per unit of delta
        for l in range(S):
            r = ucpu_row(l, -1.0); r[iU + l] = 1.0; r[ib + l] = 1.0; rows.append((r, -np.inf, 1.0))     # U <= ucpu + (1-b)
            r = umem_row(l, -1.0); r[iU + l] = 1.0; r[ib + l] = -1.0; rows.append((r, -np.inf, 0.0))    # U <= umem + b
            r = {iU + l: 1.0}; r.update({idel + l * K + s: -1.0 / K for s in range(K)}); rows.append((r, 0.0, 0.0))
            r = {iF + l: 1.0}; r.update({idel + l * K + s: -fs_ for s, fs_ in enumerate(np.diff(fs))}); rows.append((r, 0.0, 0.0))
            for s in range(K - 1):                               # delta_{s+1} <= z_s <= delta_s
                rows.append(({idel + l * K + s + 1: 1.0, iz + l * (K - 1) + s: -1.0}, -np.inf, 0.0))
                rows.append(({iz + l * (K - 1) + s: 1.0, idel + l * K + s: -1.0}, -np.inf, 0.0))
        c[iF:iF + S] = -1.0
        integ_full = np.zeros(L.n); integ_full[:L.ny] = 1; integ_full[ib:ib + S] = 1; integ_full[iz:iz + S * (K - 1)] = 1
        res, sec = run_milp(L, rows, c, lb, ub, integ_full, time_limit)
        sign = -1.0
    out = dict(sense=sense, K=K, seconds=sec, status=int(res.status), message=res.message, time_limit=time_limit)
    if res.x is None:
        out['note'] = 'no feasible solution returned'; return out
    y = res.x[:L.ny].reshape(M, S); a = [int(v) for v in y.argmax(axis=1)]
    ref_cost, prob = gen.evaluate_B(d, a)
    f, U, ucpu, umem, viol = cp1_parts(d, a)
    out.update(assign=a, feasible=not prob, reference_cost_of_incumbent=ref_cost,
               true_cp1_of_incumbent=f, milp_objective=sign * float(res.fun),
               milp_dual_bound=None if getattr(res, 'mip_dual_bound', None) is None else sign * float(res.mip_dual_bound),
               mip_gap=getattr(res, 'mip_gap', None))
    return out


# ------------------------------------------------------------------------------------------- evaluation of stored solutions
def dense_rank(v):
    order = sorted(set(round(x, 9) for x in v.values())); return {k: order.index(round(x, 9)) + 1 for k, x in v.items()}   # 1 = best


def evaluate_stored(d):
    path = os.path.join(R, 'final_block1_solutions.json')
    js = json.load(open(path)); sols = js['solutions']
    names = list(sols)
    ra = np.array(sols[REF_KEY]['assign'])
    vals, ref_costs, rows = {}, {}, []
    for k in names:
        a = sols[k]['assign']
        cost, prob = gen.evaluate_B(d, a)
        assert not prob, f'{k}: reference checker reports {prob}'
        ref_costs[k] = cost; vals[k] = cp1_objective(d, a)
    rank_ref, rank_cp1 = dense_rank(ref_costs), dense_rank(vals)
    for k in names:
        f, U, ucpu, umem, viol = cp1_parts(d, sols[k]['assign'])
        sols[k].update(ref_cost=ref_costs[k], feasible=True, hamming_to_reference=int((np.array(sols[k]['assign']) != ra).sum()))
        rows.append([k, f'{ref_costs[k]:.9f}', f'{vals[k]:.9f}', rank_ref[k], rank_cp1[k], f'{cp1_cpu_only(d, sols[k]["assign"]):.9f}',
                     int((umem > ucpu + 1e-12).sum()), f'{U.max():.6f}', 'CP1'])
    with open(os.path.join(R, 'final_block1_eval.csv'), 'w', newline='') as fh:      # ONE format (see README, section Final)
        w = csv.writer(fh)
        w.writerow(['solution', 'reference_cost', 'cp1_objective', 'rank_reference', 'rank_cp1', 'cp1_cpu_only', 'servers_memory_binding', 'max_U', 'objective_used'])
        w.writerows(rows)
    with open(os.path.join(R, 'final_block1_decomposition.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['solution', 'cp1_objective', 'cp1_cpu_only', 'servers_memory_binding', 'max_U'])
        for r in rows: w.writerow([r[0], r[2], r[5], r[6], r[7]])
    json.dump(dict(z_ref=js['z_ref'], z_LP=js['z_LP'], solutions=sols), open(path, 'w'), indent=1)
    print('Evaluation of the stored solutions (results/final_block1_solutions.json):')
    for k in names:
        print(f'  {k:28s} reference {ref_costs[k]:.9f}  CP1 {vals[k]:.9f}  rank {rank_ref[k]} -> {rank_cp1[k]}  '
              f'VMs elsewhere than in the reference optimum: {sols[k]["hamming_to_reference"]}')
    ok = True
    for k, v in CONTROL.items():
        diff = vals[k] - v; flag = abs(diff) <= 5e-7; ok &= flag
        print(f'  control {k:28s} report {v:.6f}  recomputed {vals[k]:.6f}  diff {diff:+.2e}  {"OK" if flag else "DIFFERS"}')
    for k in names:
        if k != REF_KEY:
            diff = vals[k] - vals[REF_KEY]
            print(f'  {k}: CP1 minus reference-optimum CP1 = {diff:+.6f} ({100 * diff / vals[REF_KEY]:+.3f} % of the optimum value)')
    return ok, vals, sols


def run_seeds(d):
    try:
        from final_block2 import read, sa_params_from_csv
        from src.meta import ga, sa
    except ImportError as e:
        sys.exit(f'--seeds needs the team code ({e}); not available in this checkout.')
    out = []
    for label, mod, fname in (('GA', ga, 'Telgarin_runs.csv'), ('SA', sa, 'Hafizi_runs.csv')):
        params = sa_params_from_csv(read(fname)) if label == 'SA' else None
        for s_ in range(10):
            res = mod.solve(d, INST['variant'], s_, BUDGET, params=params, start=None)
            out.append([label, s_, gen.evaluate_B(d, res['solution'])[0], cp1_objective(d, res['solution'])])
    with open(os.path.join(R, 'final_block1_seeds.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['method', 'seed', 'reference_cost', 'cp1_objective']); w.writerows(out)
    for label in ('GA', 'SA'):
        r = [x for x in out if x[0] == label]; c1 = [x[3] for x in r]; rc = [x[2] for x in r]
        print(f'  {label}: reference cost range {max(rc)-min(rc):.4f}; CP1 objective {min(c1):.6f}-{max(c1):.6f} '
              f'(range {max(c1)-min(c1):.4f}); mean {np.mean(c1):.6f} std {np.std(c1, ddof=1):.6f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resolve-reference', action='store_true')
    ap.add_argument('--optimal-set', action='store_true')
    ap.add_argument('--time-limit', type=float, default=300.0, help='seconds per MILP of --optimal-set')
    ap.add_argument('--seeds', action='store_true')
    a = ap.parse_args()
    d = instance()
    print(f"S {d['S']} M {d['M']}  CPU {d['cpu_req'].sum()}/{d['cpu_cap'].sum()}  MEM {d['mem_req'].sum()}/{d['mem_cap'].sum()}  "
          f"premium {int((d['vm_required_zone'] != -1).sum())}")
    ok, vals, sols = evaluate_stored(d)
    if a.resolve_reference:
        ex = solve_reference(d); lp = solve_reference(d, relax=True)
        same = ex['assign'] == sols[REF_KEY]['assign']
        res = dict(z_ref=ex['value'], z_LP=lp['value'], seconds_integer=ex['seconds'], mip_gap=ex.get('mip_gap'),
                   forbidden_pairs=ex['forbidden_pairs'], rows=ex['rows'], variables=ex['variables'],
                   returned_assignment_equals_stored=same, returned_assignment=ex['assign'],
                   returned_reference_cost=gen.evaluate_B(d, ex['assign'])[0], returned_cp1_objective=cp1_objective(d, ex['assign']),
                   stored_reference_cost=sols[REF_KEY]['ref_cost'], stored_cp1_objective=vals[REF_KEY])
        json.dump(res, open(os.path.join(R, 'final_block1_reference_resolve.json'), 'w'), indent=1)
        print(f"\nResolved reference model: z_ref {res['z_ref']:.9f}  z_LP {res['z_LP']:.9f}  ({ex['seconds']:.1f} s, gap {ex.get('mip_gap')})")
        print(f"  returned assignment equals the stored one: {same};  CP1 value of the returned assignment {res['returned_cp1_objective']:.9f}")
    if a.optimal_set:
        zref = json.load(open(os.path.join(R, 'final_block1_solutions.json')))['z_ref']
        out = dict(z_ref=zref, tolerance=1e-6)
        for sense in ('min', 'max'):
            r = optimal_set_range(d, zref, sense, a.time_limit); out[sense] = r
            print(f"\nCP1 over the reference-optimal set, {sense}: status {r['status']} ({r['message']}), {r['seconds']:.0f} s")
            if 'true_cp1_of_incumbent' in r:
                print(f"  incumbent: true CP1 {r['true_cp1_of_incumbent']:.6f}, reference cost {r['reference_cost_of_incumbent']:.9f}, "
                      f"MILP objective {r['milp_objective']:.6f}, MILP dual bound {r['milp_dual_bound']}, gap {r['mip_gap']}")
        json.dump(out, open(os.path.join(R, 'final_block1_optimal_set.json'), 'w'), indent=1)
    if a.seeds:
        run_seeds(d)
    if not ok:
        print('\nCONTROL VALUES DIFFER: find the cause, do not adjust the report numbers to the script.'); sys.exit(1)


if __name__ == '__main__':
    main()
