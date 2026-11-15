"""Final, Block 3 check: cost of the own (greedy) start of SA against the final cost of the SA run, per seed.
  python final/src/sa_start_check.py   -> results/final_SA_start_check.csv
The own start of sa.solve is sa.initial_solution(d, random.Random(seed)); the final costs are read from the results files."""
import csv, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
from src.meta import sa
R = os.path.join(ROOT, 'results')
fin = {('own', int(r['seed'])): float(r['cost']) for r in csv.DictReader(open(os.path.join(R, 'Hafizi_runs.csv')))}
for r in csv.DictReader(open(os.path.join(R, 'final_SA_scaling.csv'))):
    if r['start'] == 'own': fin[(r['size'], int(r['seed']))] = float(r['cost'])
with open(os.path.join(R, 'final_SA_start_check.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['instance', 'seed', 'start_cost', 'final_cost', 'improved_by_SA'])
    for name in ('own', '100', '200', '400'):
        d = gen.cp34_instance(10, 'B', 2) if name == 'own' else gen.scaled_instance('B', int(name), 2)
        n = 0; sc = []
        for seed in range(10):
            c, pr = gen.evaluate_B(d, sa.initial_solution(d, random.Random(seed)))
            f = fin[(name, seed)]; n += f < c - 1e-9; sc.append(c)
            w.writerow([name, seed, f'{c:.9f}', f'{f:.9f}', f < c - 1e-9])
        print(name, 'start mean', round(sum(sc) / 10, 4), 'min', round(min(sc), 4), 'final mean', round(sum(fin[(name, s)] for s in range(10)) / 10, 4), 'runs improved by SA', n)
