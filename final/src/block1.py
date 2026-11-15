"""Final, Block 1: the reference optimum and the best solution of the approximate methods under the Checkpoint 1 objective
    f_CP1 = sum_j (U_j + U_j^2),  U_j = max(CPU utilisation, memory utilisation) of server j.
  python final/src/block1.py     -> results/final_block1.csv
Needs final/starts/exact_B_own.json (run_exact.py --own), results/final_solutions_GA.json and final_solutions_SA.json (traces.py);
reads results/final_block1_solutions.json (final_block1.py) if present: a second optimal placement of the reference model."""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
d = gen.cp34_instance(10, 'B', 2)
S, M = d['S'], d['M']
p = np.asarray(d['cpu_req'], float); q = np.asarray(d['mem_req'], float)
C = np.asarray(d['cpu_cap'], float); Q = np.asarray(d['mem_cap'], float)
allowed = np.ones((M, S), bool)
for k in range(M):
    if d['vm_required_zone'][k] != -1:
        allowed[k] = np.asarray(d['zone']) == d['vm_required_zone'][k]


def cp1(assign):
    a = np.asarray(assign)
    u = np.maximum(np.bincount(a, weights=p, minlength=S) / C, np.bincount(a, weights=q, minlength=S) / Q)
    return float((u + u * u).sum())


def feasible(assign):
    a = np.asarray(assign)
    return bool((np.bincount(a, weights=p, minlength=S) <= C + 1e-9).all() and (np.bincount(a, weights=q, minlength=S) <= Q + 1e-9).all()
                and allowed[np.arange(M), a].all())


def descent(assign):
    """Steepest descent under f_CP1 over feasible transfers and swaps (deterministic); returns a local optimum."""
    a = list(assign); cur = cp1(a); steps = 0
    while True:
        best, mv = cur - 1e-12, None
        for k in range(M):
            old = a[k]
            for s in range(S):
                if s != old and allowed[k, s]:
                    a[k] = s
                    if feasible(a):
                        v = cp1(a)
                        if v < best: best, mv = v, ('t', k, s)
            a[k] = old
        for k in range(M):
            for l in range(k + 1, M):
                if a[k] != a[l] and allowed[k, a[l]] and allowed[l, a[k]]:
                    a[k], a[l] = a[l], a[k]
                    if feasible(a):
                        v = cp1(a)
                        if v < best: best, mv = v, ('s', k, l)
                    a[k], a[l] = a[l], a[k]
        if mv is None:
            return a, cur, steps
        if mv[0] == 't': a[mv[1]] = mv[2]
        else: a[mv[1]], a[mv[2]] = a[mv[2]], a[mv[1]]
        cur = best; steps += 1


exact = json.load(open(os.path.join(ROOT, 'final', 'starts', 'exact_B_own.json')))['assign']
ga = json.load(open(os.path.join(ROOT, 'results', 'final_solutions_GA.json')))
rows = [('reference optimum, placement A (HiGHS run of final/src/run_exact.py)', exact)]
R = os.path.join(ROOT, 'results')
if os.path.exists(os.path.join(R, 'final_block1_solutions.json')):      # files written by final/src/final_block1.py (Telgarin)
    rows.append(('reference optimum, placement B (first solve of final_block1.py)', json.load(open(os.path.join(R, 'final_block1_solutions.json')))['solutions']['Reference optimum (exact)']['assign']))
if os.path.exists(os.path.join(R, 'final_block1_reference_resolve.json')):
    rows.append(('reference optimum, placement C (second solve of final_block1.py)', json.load(open(os.path.join(R, 'final_block1_reference_resolve.json')))['returned_assignment']))
if os.path.exists(os.path.join(R, 'final_block1_optimal_set.json')):
    os_ = json.load(open(os.path.join(R, 'final_block1_optimal_set.json')))
    rows.append(('reference-optimal placement with the lowest f_CP1 found (final_block1.py, MILP, 600 s)', os_['min']['assign']))
    rows.append(('reference-optimal placement with the highest f_CP1 found (final_block1.py, MILP, 600 s)', os_['max']['assign']))
best_seed = min(ga, key=lambda s: (ga[s]['cost'], int(s)))
rows.append((f'best approximate solution (GA, seed {best_seed})', ga[best_seed]['assign']))
sa_sol = json.load(open(os.path.join(ROOT, 'results', 'final_solutions_SA.json')))
sbest = min(sa_sol, key=lambda s: (sa_sol[s]['cost'], int(s)))
rows.append((f'best SA solution (seed {sbest})', sa_sol[sbest]['assign']))
for s in sorted(ga, key=int):
    rows.append((f'GA seed {s}', ga[s]['assign']))
loc, v, steps = descent(exact)
rows.append((f'descent under f_CP1 from the reference optimum ({steps} steps)', loc))
loc2, v2, steps2 = descent(ga[best_seed]['assign'])
rows.append((f'descent under f_CP1 from the best GA solution ({steps2} steps)', loc2))
with open(os.path.join(ROOT, 'results', 'final_block1.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['solution', 'reference_objective', 'cp1_objective', 'feasible', 'servers_where_memory_util_exceeds_cpu_util', 'hamming_to_reference_optimum'])
    for name, a in rows:
        z, pr = gen.evaluate_B(d, a)
        aa = np.asarray(a)
        mem_dom = int((np.bincount(aa, weights=q, minlength=S) / Q > np.bincount(aa, weights=p, minlength=S) / C).sum())
        w.writerow([name, f'{z:.9f}', f'{cp1(a):.9f}', not pr, mem_dom, int((aa != np.asarray(exact)).sum())])
        print(name, round(z, 6), round(cp1(a), 6), not pr, mem_dom)
