"""Final report, Block 2: computations from the two SA files of Checkpoint 4 (team T10, cp34_instance(10, 'B', 2)).

Input  results/Hafizi_runs.csv     SA-I, 10 runs, seeds 0-9, budget 100000, tuned parameters (the SA row of Table 2)
       results/Shaken_tuning.csv   SA-T: random baseline of the SA pair (seeds 0-9), 80 Nelder-Mead calls x 5 tuning seeds (100-104),
                                   'compare-rule' and 'compare-tuned' (seeds 0-9)
       results/extra_scan_all_seeds.json   final SA costs re-run for the scan (cross-check of Hafizi_runs.csv)
       results/final_block1_solutions.json z_ref, z_LP
Output results/final_sa_files.json, results/final_sa_table.csv
Usage  python final/src/final_sa_files.py
"""
import csv, json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, 'results')
sol = json.load(open(os.path.join(R, 'final_block1_solutions.json')))
Z_REF, Z_LP = sol['z_ref'], sol['z_LP']


def read(name):
    with open(os.path.join(R, name), newline='') as fh:
        return list(csv.DictReader(fh))


def col(x):
    x = np.asarray(x, float)
    return dict(n=len(x), best=float(x.min()), mean=float(x.mean()), worst=float(x.max()), std=float(x.std(ddof=1)),
                gap_ref_mean=100 * (x.mean() - Z_REF) / Z_REF, gap_ref_worst=100 * (x.max() - Z_REF) / Z_REF,
                gap_LP_mean=100 * (x.mean() - Z_LP) / x.mean(), gap_LP_worst=100 * (x.max() - Z_LP) / x.max())


def cfg(rows, name):
    return [r for r in rows if r['configuration'] == name]


def main():
    H, T = read('Hafizi_runs.csv'), read('Shaken_tuning.csv')
    out = dict(z_ref=Z_REF, z_LP=Z_LP)
    # ---- 1. SA row of Table 2 from Hafizi_runs.csv
    h = [float(r['cost']) for r in H]
    scan = json.load(open(os.path.join(R, 'extra_scan_all_seeds.json')))
    assert [int(r['seed']) for r in H] == list(range(10)) and [r['seed'] for r in scan['SA']] == list(range(10))
    out['SA_row'] = col(h)
    out['SA_row'].update(feasible=f"{sum(r['feasible'] == 'True' for r in H)}/{len(H)}", mean_seconds=float(np.mean([float(r['seconds']) for r in H])),
                         budget=sorted({int(r['budget']) for r in H}), evals=sorted({int(r['evals']) for r in H}), params=json.loads(H[0]['params']),
                         max_abs_diff_to_rerun_in_scan=float(max(abs(a - b['cost']) for a, b in zip(h, scan['SA']))),
                         distinct_final_costs=len(set(h)), tied_seeds=[r['seed'] for r in H if r['cost'] == H[0]['cost']])
    # ---- 2. random baseline of the SA pair
    rnd = cfg(T, 'random')
    out['random_SA_pair'] = col([float(r['cost']) for r in rnd])
    out['random_SA_pair'].update(feasible=f"{sum(r['feasible'] == 'True' for r in rnd)}/{len(rnd)}",
                                 mean_seconds=float(np.mean([float(r['seconds']) for r in rnd])),
                                 mean_feasible_candidates=float(np.mean([json.loads(r['params'])['n_feasible_candidates'] for r in rnd])),
                                 method=json.loads(rnd[0]['params'])['method'])
    # ---- 3. tuning: 80 Nelder-Mead calls on tuning seeds 100-104, then comparison on seeds 0-9
    calls = sorted({r['configuration'] for r in T if r['configuration'].startswith('tune-NM-call')})
    cm = []
    for c in calls:
        rr = cfg(T, c); p = json.loads(rr[0]['params'])
        cm.append((c, float(np.mean([float(r['cost']) for r in rr])), p['T0'], p['Tend'], len(rr)))
    best = min(cm, key=lambda t: t[1]); first = cm[0]
    tseeds = sorted({int(r['seed']) for r in T if r['configuration'].startswith('tune-NM-call')})
    first_costs = [float(r['cost']) for r in cfg(T, 'tune-NM-call01')]
    last_costs = [float(r['cost']) for r in cfg(T, best[0])]
    out['tuning'] = dict(n_calls=len(calls), tuning_seeds=tseeds, runs_per_call=sorted({c[4] for c in cm}), budget_per_run=100000,
                         total_runs=sum(c[4] for c in cm), first_call=dict(zip(['call', 'mean', 'T0', 'Tend'], first[:4])),
                         best_call=dict(zip(['call', 'mean', 'T0', 'Tend'], best[:4])), calls_with_the_best_mean=sum(abs(c[1] - best[1]) < 1e-12 for c in cm),
                         tuning_seed_std_first_call=float(np.std(first_costs, ddof=1)), tuning_seed_se_first_call=float(np.std(first_costs, ddof=1) / np.sqrt(5)),
                         tuning_seeds_improvement_first_to_best=first[1] - best[1])
    rule, tuned = [float(r['cost']) for r in cfg(T, 'compare-rule')], [float(r['cost']) for r in cfg(T, 'compare-tuned')]
    assert [int(r['seed']) for r in cfg(T, 'compare-tuned')] == list(range(10))
    assert np.allclose(tuned, h, atol=1e-8), 'compare-tuned must equal Hafizi_runs.csv'
    out['compare_seeds_0_9'] = dict(rule=col(rule), tuned=col(tuned),
        rule_params=json.loads(cfg(T, 'compare-rule')[0]['params']), tuned_params=json.loads(cfg(T, 'compare-tuned')[0]['params']),
        tuned_equals_hafizi_runs=True, mean_diff_tuned_minus_rule=float(np.mean(tuned) - np.mean(rule)),
        welch_se=float(np.sqrt(np.var(tuned, ddof=1) / 10 + np.var(rule, ddof=1) / 10)),
        welch_t_p=float(stats.ttest_ind(tuned, rule, equal_var=False).pvalue),
        mann_whitney_p=float(stats.mannwhitneyu(tuned, rule, alternative='two-sided').pvalue),
        rule_better_seeds_nominal_pairing=int((np.array(rule) < np.array(tuned)).sum()),
        runs_reaching_zref=dict(rule=int((np.abs(np.array(rule) - Z_REF) < 1e-6).sum()), tuned=int((np.abs(np.array(tuned) - Z_REF) < 1e-6).sum())),
        tuned_equals_rule_params_of_first_call=bool(abs(first[2] - json.loads(cfg(T, 'compare-rule')[0]['params'])['T0']) < 1e-12))
    # ---- 4. Table row for the report
    with open(os.path.join(R, 'final_sa_table.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['row', 'n', 'best', 'mean', 'worst', 'std', 'gap_ref_mean', 'gap_ref_worst', 'gap_LP_mean', 'gap_LP_worst', 'source'])
        for name, d, src in (('SA, tuned parameters (Table 2 row)', out['SA_row'], 'Hafizi_runs.csv'), ('SA, rule parameters (seeds 0-9)', col(rule), 'Shaken_tuning.csv compare-rule'),
                             ('random search of the SA pair', out['random_SA_pair'], 'Shaken_tuning.csv random')):
            w.writerow([name, d['n'], *(f'{d[k]:.6f}' for k in ('best', 'mean', 'worst', 'std')), *(f'{d[k]:.4f}' for k in ('gap_ref_mean', 'gap_ref_worst', 'gap_LP_mean', 'gap_LP_worst')), src])
    json.dump(out, open(os.path.join(R, 'final_sa_files.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
