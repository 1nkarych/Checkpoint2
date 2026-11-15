"""Final: every table and figure of the report from the files in results/.
  python final/src/make_tables.py
Reads  results/Telgarin_runs.csv, Telgarin_random.csv, ref_values.json, final_SA_summary.csv, final_trace_*.csv,
       final_GA_repair.csv, final_exact.csv, final_block1.csv
Writes results/final_table2.csv, results/final_table3_report.csv (Table 3; results/final_table3.csv is the file of Hafizi's script), report/final_F1.pdf, report/final_F2.pdf and
       report/final_numbers.tex (LaTeX macros with the numbers used in the text)."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, 'results'); REP = os.path.join(ROOT, 'report')
rd = lambda n: list(csv.DictReader(open(os.path.join(R, n))))
ref = json.load(open(os.path.join(R, 'ref_values.json'))); zref, zlp = ref['z_ref'], ref['z_LP']
gr = lambda z: 100 * (z - zref) / zref
gl = lambda z: 100 * (z - zlp) / z

# ---------------- Table 2 (layout of Telgarin: gap columns = mean / worst)
def row(name, rows, note_feas='', note_time=''):
    c = np.array([float(r['cost']) for r in rows if r['feasible'] == 'True'])
    m, w = c.mean(), c.max()
    return [name, f'{c.min():.4f}', f'{m:.4f}', f'{w:.4f}', f'{c.std(ddof=1):.4f}', f'{len(c)}/{len(rows)}' + note_feas,
            f"{np.mean([float(r['seconds']) for r in rows]):.1f}" + note_time, f'{gr(m):.2f} / {gr(w):.2f}', f'{gl(m):.2f} / {gl(w):.2f}']
tun = rd('Shaken_tuning.csv')
res = json.load(open(os.path.join(R, 'final_block1_reference_resolve.json')))
t2 = [row('GA with repair ($N=30$)', rd('Telgarin_runs.csv'), '$^{*}$', '$^{*}$'),
      row('SA, transfer (tuned, CP4)', rd('Hafizi_runs.csv')),
      row('SA, transfer (rule parameters)', [r for r in tun if r['configuration'] == 'compare-rule'], '', '$^{**}$'),
      row('Random search, uniform (SA pair)', [r for r in tun if r['configuration'] == 'random']),
      row('Random search, repaired (GA pair)', rd('Telgarin_random.csv')),
      ['Exact solver (HiGHS)', f"{res['z_ref']:.4f}", f"{res['z_ref']:.4f}", f"{res['z_ref']:.4f}", '--', '1 run, proven', f"{res['seconds_integer']:.1f}", f"{gr(res['z_ref']):.2f}", f"{gl(res['z_ref']):.2f}"]]
with open(os.path.join(R, 'final_table2.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['method', 'best', 'mean', 'worst', 'std', 'feasible', 'time_s', 'gap_ref_pct_mean_worst', 'gap_LP_pct_mean_worst'])
    for r in t2: w.writerow([x.replace('$^{*}$', '').replace('$^{**}$', '').replace('$N=30$', 'N=30') for x in r])
with open(os.path.join(REP, 'final_table2.tex'), 'w') as fh:
    for r in t2: fh.write(' & '.join(r) + ' \\\\\n')
for r in t2: print(r)

# ---------------- Figure F1
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for name, lab, col in (('GA', 'GA, repair', 'C0'), ('SA', 'SA, transfer only', 'C2'), ('random', 'Random search', 'C1')):
    f = os.path.join(R, f'final_trace_{name}.csv')
    if not os.path.exists(f): continue
    tr = rd(f'final_trace_{name}.csv'); seeds = sorted({int(r['seed']) for r in tr})
    grid = sorted({int(r['evals']) for r in tr})
    grid = [e for e in grid if all(any(int(r['evals']) == e and int(r['seed']) == s for r in tr) for s in seeds[:1])]
    per = {s: dict((int(r['evals']), float(r['best'])) for r in tr if int(r['seed']) == s) for s in seeds}
    grid = [e for e in grid if all(e in per[s] and np.isfinite(per[s][e]) for s in seeds)]
    Y = np.array([[per[s][e] for e in grid] for s in seeds])
    ax.plot(grid, Y.mean(0), color=col, label=f'{lab} (mean of {len(seeds)} seeds, band min-max)')
    ax.fill_between(grid, Y.min(0), Y.max(0), color=col, alpha=0.2, lw=0)
if not os.path.exists(os.path.join(R, 'final_trace_SA.csv')):
    pts = [r for r in rd('final_SA_summary.csv') if r['configuration'] == 'SA curve point']
    ax.plot([int(r['evals']) for r in pts], [float(r['mean']) for r in pts], 's', color='C2', ms=5,
            label='SA, transfer only (mean of 10 seeds; points reported in Checkpoint 4)')
ax.axhline(zref, color='k', ls='--', lw=1, label=f'$z_{{ref}}$ = {zref:.4f}')
ax.axhline(zlp, color='gray', ls=':', lw=1.2, label=f'$z_{{LP}}$ = {zlp:.4f}')
ax.set_xscale('log'); ax.set_xlim(100, 100000); ax.set_ylim(8.3, 9.4)
ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost so far'); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(REP, 'final_F1.pdf')); plt.close(fig)

# ---------------- Table 3: run of Hafizi (final_scaling_*.csv); best known = best over ALL runs of the team (also the second run)
ex = {r['instance']: r for r in rd('final_exact.csv')}                      # second run (other machine)
exH = {r['size']: r for r in rd('final_scaling_exact.csv')}                 # run of Table 3
runsB = rd('final_GA_repair.csv') + rd('final_SA_scaling.csv')
runsH = rd('final_scaling_methods.csv')
t3 = []
for size in ('100', '200', '400'):
    e = exH[size]; zo = float(e['objective'])
    allc = [zo, float(ex[size]['objective'])] + [float(r['cost']) for r in runsB + runsH if r['size'] == size and r['feasible'] == 'True']
    best_known = min(allc)
    bk = lambda v: 100 * (v - best_known) / best_known
    t3.append([size, 'exact (HiGHS)', zo, zo, best_known, bk(zo), bk(zo), float(e['solver_gap_pct']), e['proven'], float(e['seconds']), '1/1', float(e['best_bound']), ''])
    for m, lab in (('GA repair', 'GA'), ('SA transfer', 'SA')):
        for st, sl in (('own', 'own start'), ('LP-rounded', 'LP-rounded start')):
            rr = [r for r in runsH if r['size'] == size and r['method'] == m and r['start'] == st]
            c = np.array([float(r['cost']) for r in rr if r['feasible'] == 'True'])
            t3.append([size, f'{lab}, {sl}', c.min(), c.mean(), best_known, bk(c.min()), bk(c.mean()), '', '', np.mean([float(r['seconds']) for r in rr]), f'{len(c)}/{len(rr)}', c.std(ddof=1), c.max()])
with open(os.path.join(R, 'final_table3_report.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['size', 'method', 'best', 'mean', 'best_known_all_runs', 'gap_best_pct', 'gap_mean_pct', 'solver_gap_pct', 'proven', 'time_s', 'feasible', 'bound_or_std', 'worst'])
    for r in t3: w.writerow([f'{v:.6f}' if isinstance(v, float) else v for v in r])
with open(os.path.join(REP, 'final_table3.tex'), 'w') as fh:
    for r in t3:
        if r[1].startswith('exact'):
            fh.write(('\\midrule\n' if r[0] != '100' else '') + f"$M={r[0]}$ & exact (HiGHS) & {r[2]:.4f} & -- & {r[4]:.4f} & {r[5]:.2f} & -- & {r[7]:.2f} ({'proven' if r[8] == 'True' else 'not proven'}) & {r[9]:.1f}\\\\\n")
        else:
            fh.write(f" & {r[1]} & {r[2]:.4f} & {r[3]:.4f} & {r[4]:.4f} & {r[5]:.2f} & {r[6]:.2f} & & {r[9]:.1f}\\\\\n")
for r in t3: print([round(v, 4) if isinstance(v, float) else v for v in r])

# ---------------- Figure F2
fig, ax = plt.subplots(figsize=(5.8, 3.5))
xs = [100, 200, 400]; ys = [float(exH[str(k)]['seconds']) for k in xs]; pr = [exH[str(k)]['proven'] == 'True' for k in xs]
ax.plot(xs, ys, '-', color='C0', lw=1.2, label='run of Table 3')
xb = [50, 100, 200, 400]; yb = [float(ex[k]['seconds']) for k in ('own', '100', '200', '400')]; pb = [ex[k]['proven'] == 'True' for k in ('own', '100', '200', '400')]
ax.plot(xb, yb, '--', color='gray', lw=1, label='second run, other machine')
for X, Y, P, col in ((xs, ys, pr, 'C0'), (xb, yb, pb, 'gray')):
    ax.plot([x for x, q in zip(X, P) if q], [y for y, q in zip(Y, P) if q], 'o', color=col)
    ax.plot([x for x, q in zip(X, P) if not q], [y for y, q in zip(Y, P) if not q], 'x', color='C3', ms=9, mew=2)
ax.plot([], [], 'o', color='k', label='optimum proven (solver gap 0)'); ax.plot([], [], 'x', color='C3', mew=2, label='stopped by the limit, not proven')
ax.axhline(120, color='C3', ls=':', lw=1, label='limit 120 s')
for x, y in zip(xs[:1], ys[:1]): ax.annotate(f'{y:.1f} s', (x, y), textcoords='offset points', xytext=(6, -12), fontsize=8)
ax.annotate(f'{yb[0]:.1f} s', (xb[0], yb[0]), textcoords='offset points', xytext=(6, 2), fontsize=8); ax.annotate(f'{yb[1]:.1f} s', (xb[1], yb[1]), textcoords='offset points', xytext=(-34, 4), fontsize=8)
ax.set_xlabel('M (virtual machines); M = 50 is the own instance'); ax.set_ylabel('time of HiGHS, s'); ax.set_xticks(xb)
ax.set_ylim(0, 135); ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(REP, 'final_F2.pdf')); plt.close(fig)
