"""Evaluate the three T10 solutions under the Checkpoint 1 objective f = sum_j (U_j + U_j^2), U_j = max(U_cpu, U_mem)."""
import json, csv, numpy as np
import checkpoint4_generator_PUBLIC as gen
d = gen.cp34_instance(10, 'B', 2)
sols = json.load(open('results/final_block1_solutions.json'))['solutions']
print('S', d['S'], 'M', d['M'], 'CPU', d['cpu_req'].sum(), '/', d['cpu_cap'].sum(), 'MEM', d['mem_req'].sum(), '/', d['mem_cap'].sum(),
      'premium', int((d['vm_required_zone'] != -1).sum()))
def parts(a):
    a = np.asarray(a); S = d['S']
    ucpu = np.bincount(a, weights=d['cpu_req'], minlength=S) / d['cpu_cap']
    umem = np.bincount(a, weights=d['mem_req'], minlength=S) / d['mem_cap']
    U = np.maximum(ucpu, umem)
    return float((U + U**2).sum()), float((ucpu + ucpu**2).sum()), ucpu, umem, U
rows = []
for k, v in sols.items():
    ref, prob = gen.evaluate_B(d, v['assign'])
    f, fcpu, ucpu, umem, U = parts(v['assign'])
    nmem = int((umem > ucpu + 1e-12).sum())
    print(f'{k:28s} ref {ref:.9f} feas {not prob} | CP1 {f:.9f} | CP1 cpu-only {fcpu:.9f} | mem-binding servers {nmem} | maxU {U.max():.4f}')
    rows.append([k, ref, f, fcpu, nmem])
    v['_u'] = (ucpu, umem)
with open('results/final_block1_eval.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['solution', 'reference_cost', 'cp1_objective', 'cp1_cpu_only', 'servers_memory_binding']); w.writerows(rows)
for k, v in sols.items():
    uc, um = v['_u']; print(k); print('  cpu', np.round(uc, 4)); print('  mem', np.round(um, 4))
