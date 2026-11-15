"""Final report, Block 1: from my Checkpoint 1 model to the reference model (team T10, cp34_instance(10, 'B', 2)).

The instruction asks for ONE computation on your own data: take the optimal solution of the reference model (value z_ref)
and the best solution found by the approximate methods, and evaluate both under YOUR OWN Checkpoint 1 objective.
Does the order of the two solutions stay the same?

This script does the part that does not depend on your Checkpoint 1 model:
  1. solves the reference model exactly (HiGHS) and keeps the optimal assignment,
  2. reruns the best seed of every method of the team (GA seed from Telgarin_runs.csv, SA seed from Hafizi_runs.csv;
     same seed -> same run) and keeps the best assignment,
  3. writes them to results/final_block1_solutions.json,
  4. evaluates every solution under (a) the reference cost (the checker) and (b) YOUR objective.

YOUR objective goes into cp1_objective(d, assign) below. Each member has his or her own Checkpoint 1 model
(the module that gave cloud_variant(seed, variant)); the script cannot guess it. Until you fill it in, the script stops
at step 4. Run with --example to see the output format with a ILLUSTRATIVE objective (the exact latency u/(1-u) without
the piecewise-linear approximation). Do NOT put the example numbers into the report.

Usage (repository root):
    python final/src/final_block1.py              # steps 1-3, then step 4 with your objective
    python final/src/final_block1.py --example    # step 4 with the illustrative objective
Writes: results/final_block1_solutions.json, results/final_block1_eval.csv
"""
import argparse, csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint4_generator_PUBLIC as gen
from final_block2 import reference_values, read, sa_params_from_csv
from src.meta import ga, sa
from src.meta.interface import check

R = os.path.join(ROOT, 'results')
INST = dict(seed=10, story='B', variant=2)
BUDGET = 100000


def cpu_utilisation(d, assign):
    cpu = np.zeros(d['S'])
    for k, l in enumerate(assign):
        cpu[int(l)] += d['cpu_req'][k]
    return cpu / d['cpu_cap']


# ---------------------------------------------------------------------------------------------------------------
# FILL IN: your Checkpoint 1 objective, evaluated on an assignment `assign` (assign[k] = server of VM k) of the instance d.
# Use the same quantities as in your Checkpoint 1 report (for example your latency function, your penalty for imbalance,
# a cost per server, ...). If your objective contains a part that the reference model does not have (for example
# migration cost, energy, a second resource), say in the report which part it is and what you did instead.
# ---------------------------------------------------------------------------------------------------------------
def cp1_parts(d, assign):
    """Checkpoint 1 model of T10 (T10_Checkpoint1.pdf): f(x) = sum_j (U_j + U_j^2), U_j = max(U_cpu_j, U_mem_j).
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
    """Decomposition helper (NOT the CP1 objective): sum_j (u_j + u_j^2) with CPU utilisation only, as in the reference latency."""
    f, U, ucpu, umem, viol = cp1_parts(d, assign)
    return float(np.sum(ucpu + ucpu ** 2))


def example_objective(d, assign):
    """ILLUSTRATIVE ONLY: sum over servers of u/(1-u), the exact convex latency of which the reference model uses tangents."""
    u = cpu_utilisation(d, assign)
    return float(np.sum(u / (1.0 - u)))


def best_run(d, module, rows, label):
    best = min(rows, key=lambda r: float(r['cost']))
    seed = int(best['seed'])
    params = sa_params_from_csv(rows) if label == 'SA' else None
    res = module.solve(d, INST['variant'], seed, BUDGET, params=params, start=None)
    cost, pr = check(d, INST['variant'], res['solution'])
    assert not pr and abs(cost - float(best['cost'])) < 1e-6, f'{label}: rerun of seed {seed} does not reproduce the CSV'
    return seed, [int(v) for v in res['solution']], cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--example', action='store_true', help='use the illustrative objective instead of cp1_objective')
    ap.add_argument('--seeds', action='store_true', help='rerun seeds 0-9 of GA and SA and report the spread under the CP1 objective')
    a = ap.parse_args()
    d = gen.cp34_instance(INST['seed'], INST['story'], INST['variant'])

    ref = reference_values(d)
    sols = {'Reference optimum (exact)': dict(assign=ref['assign'], seed=None)}
    for label, mod, fname in (('GA best seed', ga, 'Telgarin_runs.csv'), ('SA best seed', sa, 'Hafizi_runs.csv')):
        rows = read(fname)
        seed, assign, cost = best_run(d, mod, rows, label[:2])
        sols[label] = dict(assign=assign, seed=seed)
        print(f'{label}: seed {seed}, reference cost {cost:.9f} (reproduces the CSV)')
    ra = np.array(sols['Reference optimum (exact)']['assign'])
    for k, v in sols.items():
        v['ref_cost'] = gen.evaluate_B(d, v['assign'])[0]
        v['feasible'] = not gen.evaluate_B(d, v['assign'])[1]
        v['hamming_to_reference'] = int((np.array(v['assign']) != ra).sum())
        v['cpu_utilisation_min_max'] = [float(cpu_utilisation(d, v['assign']).min()), float(cpu_utilisation(d, v['assign']).max())]
    json.dump(dict(z_ref=ref['z_ref'], z_LP=ref['z_LP'], solutions=sols),
              open(os.path.join(R, 'final_block1_solutions.json'), 'w'), indent=1)
    print('Written results/final_block1_solutions.json')
    for k, v in sols.items():
        print(f'  {k:28s} ref cost {v["ref_cost"]:.9f}  feasible {v["feasible"]}  VMs on another server than the reference optimum: {v["hamming_to_reference"]}')

    obj = example_objective if a.example else cp1_objective
    try:
        vals = {k: obj(d, v['assign']) for k, v in sols.items()}
    except NotImplementedError as e:
        print('\nStep 4 skipped:', e); return
    names = list(sols)
    def dense_rank(v):   # equal values (up to 1e-9) share a rank: the reference optimum is not unique
        order = sorted(set(round(x, 9) for x in v.values())); return {k: order.index(round(x, 9)) for k, x in v.items()}
    rank_ref = dense_rank({k: sols[k]['ref_cost'] for k in names}); rank_cp1 = dense_rank(vals)
    tag = 'EXAMPLE objective (do not use in the report)' if a.example else 'your Checkpoint 1 objective'
    print(f'\nEvaluation under {tag}:')
    with open(os.path.join(R, 'final_block1_eval.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['solution', 'reference_cost', 'cp1_objective', 'rank_reference', 'rank_cp1', 'objective_used'])
        for k in names:
            w.writerow([k, f'{sols[k]["ref_cost"]:.9f}', f'{vals[k]:.9f}', rank_ref[k], rank_cp1[k], 'EXAMPLE' if a.example else 'CP1'])
            print(f'  {k:28s} reference {sols[k]["ref_cost"]:.6f}   cp1 {vals[k]:.6f}   rank {rank_ref[k]} -> {rank_cp1[k]}')
    ref_k = 'Reference optimum (exact)'
    for k in names:
        if k == ref_k: continue
        diff = vals[k] - vals[ref_k]
        print(f'  {k}: cp1 objective minus reference-optimum value = {diff:+.6f} ({100 * diff / abs(vals[ref_k]):+.3f} %)  -> '
              + ('order kept (reference optimum is not worse)' if diff >= -1e-9 else 'ORDER REVERSED under the cp1 objective'))

    if not a.example:
        print('\nDecomposition (which part of the CP1 objective drives the values):')
        rows = []
        for k in names:
            f, U, ucpu, umem, viol = cp1_parts(d, sols[k]['assign'])
            n_mem = int((umem > ucpu + 1e-12).sum())
            rows.append([k, f, cp1_cpu_only(d, sols[k]['assign']), n_mem, float(U.max())])
            print(f'  {k:28s} cp1 {f:.6f}  cp1 with CPU only {rows[-1][2]:.6f}  servers where memory binds: {n_mem}  max U {U.max():.4f}')
        with open(os.path.join(R, 'final_block1_decomposition.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['solution', 'cp1_objective', 'cp1_cpu_only', 'servers_memory_binding', 'max_U']); w.writerows(rows)

    if a.seeds and not a.example:
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
            print(f'  {label}: reference cost range {max(rc)-min(rc):.4f}; cp1 objective {min(c1):.6f}-{max(c1):.6f} (range {max(c1)-min(c1):.4f}); '
                  f'best seed by reference {min(r, key=lambda x: x[2])[1]}, by cp1 {min(r, key=lambda x: x[3])[1]}')


if __name__ == '__main__':
    main()
