"""Final, Block 2 (Figure F1) and Block 1: rerun of the Checkpoint 4 final runs on the own instance (same seed -> same run)
to store the best-so-far traces and the final solutions.
  python final/src/traces.py --module src.meta.ga --name GA
  python final/src/traces.py --module src.meta.random_search --name random
Writes results/final_trace_<name>.csv (seed, evals, best) and results/final_solutions_<name>.json."""
import argparse, csv, importlib, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
ap = argparse.ArgumentParser(); ap.add_argument('--module', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--params', default=None)
a = ap.parse_args()
d = gen.cp34_instance(10, 'B', 2)
solve = importlib.import_module(a.module).solve
params = json.loads(a.params) if a.params else None
sols = {}
with open(os.path.join(ROOT, 'results', f'final_trace_{a.name}.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['method', 'seed', 'evals', 'best'])
    for seed in range(10):
        t0 = time.time()
        res = solve(d, 2, seed, 100000, params=params)
        cost, problems = gen.evaluate_B(d, res['solution'])
        sols[seed] = dict(cost=cost, feasible=not problems, seconds=round(time.time() - t0, 2), assign=[int(v) for v in res['solution']])
        for e, b in res['trace']:
            if e % 500 == 0 or e <= 2000:
                w.writerow([a.name, seed, e, f'{b:.9f}'])
        fh.flush(); print(a.name, seed, cost, flush=True)
json.dump(sols, open(os.path.join(ROOT, 'results', f'final_solutions_{a.name}.json'), 'w'))
