"""Final report, Block 2: scan of the final solution of EVERY seed (0-9) of GA and SA for improving transfers / swaps, and
best-improvement descent (transfers only, transfers + swaps) from each final point. Part of the final pipeline: it produces
results/extra_scan_all_seeds.json, which final/src/final_stats.py reads (mechanism 1 and 2 of the analysis).

Needs the team code: final_block2.scan_neighbourhoods, final_block2.read, final_block2.sa_params_from_csv, src.meta (ga, sa)
and the run CSVs of Checkpoint 4. These are not part of the working copy in which this file was moved to final/src/, so the
script could NOT be re-run there; the JSON in results/ is the file produced earlier by the same code (unchanged).
Usage (repository root):  python final/src/extra_scan.py     (about 3 min)
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint4_generator_PUBLIC as gen
from final_block2 import scan_neighbourhoods, sa_params_from_csv, read
from src.meta import ga, sa
d = gen.cp34_instance(10, 'B', 2); ZREF = 8.382826336851886
def descent(assign, use_swap=True):
    a = list(assign); cost = gen.evaluate_B(d, a)[0]
    while True:
        best = (-1e-9, None)
        for i in range(d['M']):
            for j in range(d['S']):
                if a[i] == j: continue
                c = list(a); c[i] = j; v, p = gen.evaluate_B(d, c)
                if not p and v - cost < best[0]: best = (v - cost, c)
        if use_swap:
            for i in range(d['M']):
                for k in range(i + 1, d['M']):
                    if a[i] == a[k]: continue
                    c = list(a); c[i], c[k] = c[k], c[i]; v, p = gen.evaluate_B(d, c)
                    if not p and v - cost < best[0]: best = (v - cost, c)
        if best[1] is None: return a, cost
        a, cost = best[1], cost + best[0]
out = {}
for name, mod, f in (('SA', sa, 'Hafizi_runs.csv'), ('GA', ga, 'Telgarin_runs.csv')):
    params = sa_params_from_csv(read(f)) if name == 'SA' else None
    rows = []
    for s in range(10):
        r = mod.solve(d, 2, s, 100000, params=params, start=None)
        sc = scan_neighbourhoods(d, r['solution'])
        _, c_sw = descent(r['solution'], True)
        _, c_tr = descent(r['solution'], False)
        rows.append(dict(seed=s, cost=r['cost'], tr_improving=sc['transfers']['improving'], sw_improving=sc['swaps']['improving'],
                         best_swap_delta=sc['swaps']['best_delta'], descent_transfer_only=c_tr, descent_transfer_swap=c_sw))
        print(name, rows[-1], flush=True)
    out[name] = rows
json.dump(out, open(os.path.join(ROOT, 'results', 'extra_scan_all_seeds.json'), 'w'), indent=1, default=float)
