"""Final: LP relaxation of the scaled instances and the LP-rounded starts (rule: model_b.round_lp).
  python final/src/make_starts.py      -> final/starts/rounded_B_<M>.json, results/final_lp_rounding.csv"""
import csv, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint4_generator_PUBLIC as gen
from model_b import solve_lp, round_lp
rows = []
for name in ('own', 100, 200, 400):
    d = gen.cp34_instance(10, 'B', 2) if name == 'own' else gen.scaled_instance('B', name, 2)
    zlp, x = solve_lp(d)
    fr = (x > 1e-6) & (x < 1 - 1e-6)
    a = round_lp(d, x)
    cost, problems = gen.evaluate_B(d, a)
    json.dump({'assign': a}, open(os.path.join(ROOT, 'final', 'starts', f'rounded_B_{name}.json'), 'w'))
    cpu_fill = sum(d['cpu_req']) / sum(d['cpu_cap']); mem_fill = sum(d['mem_req']) / sum(d['mem_cap'])
    rows.append([name, d['S'], d['M'], f'{zlp:.9f}', int(fr.sum()), int((fr.sum(axis=0) > 0).sum()), f'{cost:.9f}', len(problems) == 0, f'{cpu_fill:.4f}', f'{mem_fill:.4f}'])
    print(rows[-1])
with open(os.path.join(ROOT, 'results', 'final_lp_rounding.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['instance', 'S', 'M', 'z_LP', 'fractional_x', 'split_vms', 'rounded_cost', 'rounded_feasible', 'cpu_fill', 'mem_fill']); w.writerows(rows)
