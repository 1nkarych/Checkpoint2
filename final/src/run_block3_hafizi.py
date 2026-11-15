#!/usr/bin/env python3
"""T10 Final / Block 3, B2. End-to-end reproducible scaling experiment.

Final: python final/src/run_block3_hafizi.py
Smoke: python final/src/run_block3_hafizi.py --smoke --sizes 100 --seeds 0 --budget 100 --time-limit 2

The production run enforces M=100/200/400, seeds 0..9, CP4 budget=100000,
HiGHS exact time limit=120 seconds, and BOTH starts for BOTH methods.
The script never labels partial/smoke results as submission-ready.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.optimize import linprog, milp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import checkpoint4_generator_PUBLIC as gen
from final.src.block3_scaled_model_hafizi import build_model
from final.src.block3_lp_rounding_hafizi import round_lp
from src.meta import sa, ga

FINAL_SIZES = (100, 200, 400)
FINAL_SEEDS = tuple(range(10))
SA_PARAMS = {'T0': 1.0570236966839537, 'Tend': 0.010473720352671537}
GA_PARAMS = {'N': 30, 'pc': 0.9, 'n_mut_max': 2, 'tournament': 2,
             'elite': 1, 'repair': True, 'rho': 100.0}
METHOD_FILES = {'SA transfer': sa.solve, 'GA repair': ga.solve}

EXACT_COLS = ['team', 'story', 'variant', 'size', 'servers', 'time_limit_s',
              'objective', 'best_bound', 'solver_gap_pct', 'proven', 'seconds',
              'has_feasible_incumbent', 'status', 'message']
LP_COLS = ['team', 'story', 'variant', 'size', 'z_LP', 'seconds', 'rounding_rule',
           'trial', 'rounded_cost', 'rounded_feasible', 'start_file', 'status']
METHOD_COLS = ['team', 'story', 'variant', 'size', 'method', 'start', 'seed',
               'budget', 'params_json', 'cost', 'feasible', 'evals', 'seconds',
               'status', 'error']
TABLE_COLS = ['size', 'method', 'start', 'value', 'best_known', 'gap_to_best_known_pct',
              'solver_gap_pct', 'seconds', 'status', 'valid_runs', 'z_LP']


def parse_seeds(s):
    if '-' in s:
        a, b = (int(x) for x in s.split('-', 1))
        return tuple(range(a, b+1))
    return tuple(int(x) for x in s.split(','))


def load_csv(p):
    if not p.exists(): return []
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def add_row(p, keys, row):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        if f.tell() == 0: w.writeheader()
        w.writerow(row)


def get_row(rows, **kwargs):
    for r in rows:
        if all(str(r.get(k,'')) == str(v) for k, v in kwargs.items()):
            return r
    return None


def ffloat(value):
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError): return None


def exact_run(model, limit_s):
    """SciPy HiGHS MILP (same matrix route as final_kit.exact_helper).

    Check the integer incumbent with the OFFICIAL evaluate_B checker. Use the
    dual bound returned by HiGHS, never its status label alone, to prove optimum.
    """
    t0 = time.perf_counter()
    result = milp(**model.milp_kwargs(), options={'time_limit': float(limit_s),
                                                  'mip_rel_gap': 1e-6, 'disp': False})
    secs = time.perf_counter()-t0
    candidate = None
    status = 'not solved within the limit'
    if result.x is not None:
        M, S = model.M, model.S
        Y = np.asarray(result.x[:M*S]).reshape(M, S)
        if np.max(np.abs(Y-np.rint(Y))) <= 1e-5:
            candidate = np.argmax(Y, axis=1).astype(int).tolist()
            cost, issues = gen.evaluate_B(model.d, candidate)
            if issues or not np.isfinite(cost):
                candidate = None
                status = 'solver incumbent failed official checker'
            else:
                status = 'feasible incumbent; optimality not proven'
        else:
            status = 'returned solution not integral'
    objective = float(gen.evaluate_B(model.d, candidate)[0]) if candidate is not None else None
    bound = ffloat(getattr(result, 'mip_dual_bound', None))
    gap = (max(0.0, 100.0*(objective-bound)/abs(objective))
           if objective is not None and bound is not None and abs(objective)>1e-12 else None)
    proven = bool(candidate is not None and gap is not None and gap <= 1e-4)  # relative 1e-6 expressed in percent
    if proven: status = 'proven optimal (solver bound meets incumbent)'
    return dict(objective=objective, best_bound=bound, solver_gap_pct=gap,
                proven=proven, seconds=secs, has_feasible_incumbent=candidate is not None,
                status=status, message=str(result.message))


def try_lp_round(model, outdir, size):
    start_path = outdir / f'final_LP_rounded_B2_{size}.json'
    t0 = time.perf_counter()
    try:
        lp = linprog(**model.linprog_kwargs(), method='highs')
        lp_sec = time.perf_counter()-t0
        if not lp.success or lp.x is None: raise RuntimeError(f'LP failed: {lp.message}')
        start, meta = round_lp(model.d, lp.x)
        cost, problems = gen.evaluate_B(model.d, start)
        if problems: raise RuntimeError('Rounded start failed official checker: '+str(problems[:4]))
        start_path.write_text(json.dumps({'assign': start,
                                         'rounding_rule': meta['rounding_rule'],
                                         'LP_objective': float(lp.fun),
                                         'cost': float(cost), 'size': size}, indent=2), encoding='utf-8')
        return dict(z_LP=float(lp.fun), seconds=lp_sec, rounding_rule=meta['rounding_rule'],
                    trial=meta['trial'], rounded_cost=float(cost), rounded_feasible=True,
                    start_file=str(start_path.relative_to(ROOT)), status='ok'), start
    except Exception as ex:
        return dict(z_LP='', seconds=time.perf_counter()-t0, rounding_rule='confidence-priority LP-guided feasible greedy',
                    trial='', rounded_cost='', rounded_feasible=False,
                    start_file='', status=f'error: {ex}'), None


def fmt(x, decimals=6):
    value=ffloat(x)
    return '' if value is None else f'{value:.{decimals}f}'


def tabulate(outdir, sizes, seeds, budget, exact_limit, smoke):
    ex = load_csv(outdir/'final_scaling_exact.csv')
    lp = load_csv(outdir/'final_scaling_lp.csv')
    runs = load_csv(outdir/'final_scaling_methods.csv')
    complete = (not smoke and tuple(sizes) == FINAL_SIZES and tuple(seeds) == FINAL_SEEDS
                and budget == 100000 and exact_limit == 120)
    if complete:
        for size in sizes:
            e = get_row(ex, size=size)
            l = get_row(lp, size=size)
            if not e or not l or l.get('status') != 'ok': complete=False
            for meth in METHOD_FILES:
                for start in ('own', 'LP-rounded'):
                    for seed in seeds:
                        r = get_row(runs, size=size, method=meth, start=start, seed=seed)
                        if not r or r.get('status') != 'ok' or int(r.get('evals',0)) != budget:
                            complete = False
    table = []
    for size in sizes:
        e = get_row(ex, size=size)
        l = get_row(lp, size=size)
        zLP = l.get('z_LP', '') if l else ''
        values = []
        if e and ffloat(e.get('objective')) is not None: values.append(float(e['objective']))
        for r in runs:
            if int(r.get('size', -1)) == size and r.get('status') == 'ok' and r.get('feasible') == 'True' and ffloat(r.get('cost')) is not None:
                values.append(float(r['cost']))
        known = min(values) if values else None
        def gap(val):
            return 100*(val-known)/abs(known) if (known is not None and val is not None and abs(known)>1e-12) else None
        ev = ffloat(e.get('objective')) if e else None
        table.append(dict(size=size, method='Exact HiGHS', start='N/A', value=fmt(ev),
                          best_known=fmt(known), gap_to_best_known_pct=fmt(gap(ev), 4),
                          solver_gap_pct=fmt(e.get('solver_gap_pct'),4) if e else '',
                          seconds=fmt(e.get('seconds'), 2) if e else '',
                          status=e.get('status') if e else 'not run',
                          valid_runs='1' if ev is not None else '0', z_LP=fmt(zLP)))
        for meth in METHOD_FILES:
            for start in ('own', 'LP-rounded'):
                sub = [r for r in runs if r.get('method')==meth and r.get('start')==start
                       and int(r.get('size',-1))==size and r.get('status')=='ok' and r.get('feasible')=='True'
                       and ffloat(r.get('cost')) is not None]
                val = min((float(r['cost']) for r in sub), default=None)
                valid = len(sub)
                seconds = np.mean([float(r['seconds']) for r in sub]) if sub else None
                table.append(dict(size=size, method=meth, start=start, value=fmt(val),
                                  best_known=fmt(known), gap_to_best_known_pct=fmt(gap(val),4),
                                  solver_gap_pct='', seconds=fmt(seconds,2),
                                  status='ok' if valid==len(seeds) else f'partial ({valid}/{len(seeds)})',
                                  valid_runs=str(valid), z_LP=fmt(zLP)))
    out=outdir/('final_table3.csv' if complete else 'DRAFT_INCOMPLETE_table3.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=TABLE_COLS); w.writeheader(); w.writerows(table)
    fig, ax=plt.subplots(figsize=(8,4.7))
    xs=[]; ys=[]; annotations=[]
    for size in sizes:
        e=get_row(ex,size=size)
        if e and ffloat(e.get('seconds')) is not None:
            xs.append(size); ys.append(float(e['seconds']))
            annotations.append('proven' if e.get('proven')=='True' else 'not proven')
    if xs:
        ax.plot(xs, ys, marker='o', linewidth=1.8, label='HiGHS exact method')
        for size, y, note in zip(xs,ys,annotations):
            ax.annotate(note,(size,y),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.axhline(120, linestyle='--', linewidth=1.4, label='Official limit: 120 s')
    ax.set_xlabel('Number of virtual machines (M)')
    ax.set_ylabel('HiGHS wall-clock time (seconds)')
    ax.set_xticks(list(sizes))
    ax.set_title('Figure F2 | HiGHS runtime versus scaled instance size')
    ax.grid(True, alpha=.3); ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outdir/('final_F2.png' if complete else 'DRAFT_INCOMPLETE_F2.png'), dpi=190)
    plt.close(fig)
    write_analysis(outdir,table,ex,lp,sizes,complete,smoke)
    report=dict(complete=complete, smoke=smoke, sizes=list(sizes), seeds=list(seeds), budget=budget,
                exact_limit=exact_limit, recorded_methods=len(runs), recorded_exact=len(ex),
                table=str(out), figure='final_F2.png' if complete else 'DRAFT_INCOMPLETE_F2.png')
    (outdir/'run_manifest.json').write_text(json.dumps(report,indent=2))
    return report


def write_analysis(outdir,table,ex,lp,sizes,complete,smoke):
    lines=['# Block 3 Scaling Analysis (T10, B2)',
           '> Submission-ready: '+('YES' if complete else 'NO: partial run or smoke test; DO NOT submit these numbers.'),
           '', 'Method: CP3 binary assignment MILP, B1–B4, CPU/memory/zone constraints; HiGHS 120s. '
           'SA-transfer and GA-repair preserve their CP4 parameters, 100,000 evaluations, seeds 0–9. '
           'LP-rounded start: process VMs by LP confidence and place each on the feasible allowed '
           'server with highest fractional y; break ties by least additional latency. The rounded '
           'assignment is checked with evaluate_B. Both methods receive it through solve(start=...).', '']
    for size in sizes:
        rows=[r for r in table if int(r['size'])==size]
        exact=rows[0]
        lines.append(f'## M={size}, S={gen.scaled_instance("B",size,2)["S"]}')
        if not exact['value']:
            lines.append('HiGHS: no feasible incumbent recorded within the imposed limit.')
        else:
            lines.append(f'HiGHS: incumbent {exact["value"]}; solver gap {exact["solver_gap_pct"] or "not available"}%; '
                         f'elapsed {exact["seconds"]} s; {exact["status"]}.')
        for r in rows[1:]:
            lines.append(f'{r["method"]}, {r["start"]}: best {r["value"] or "not available"}, '
                         f'gap to best known {r["gap_to_best_known_pct"] or "N/A"}%, '
                         f'valid runs {r["valid_runs"]}.')
        if exact['value'] and 'not proven' in exact['status'].lower():
            improving=[r for r in rows[1:] if r['value'] and float(r['value']) < float(exact['value'])-1e-7]
            if improving:
                lines.append('Approximate method(s) beating the HiGHS incumbent: '+', '.join(f'{r["method"]} ({r["start"]})' for r in improving)+'.')
            else: lines.append('No approximate method in these runs beat the HiGHS incumbent.')
        lines.append('')
    lines.append('## Cross-size comparison')
    first_not_proven = next((r for r in table if r['method']=='Exact HiGHS' and
                             r['size'] in sizes and r['status'] != 'proven optimal (solver bound meets incumbent)'), None)
    if first_not_proven:
        lines.append(f'First size without a proof of optimality in these measurements: M={first_not_proven["size"]}. '
                     'A missing solver bound or incumbent must be reported as such.')
    else:
        lines.append('The exact solver proved optimality at all three tested sizes.' if complete else
                     'No failure of proof is established by the available partial runs.')
    first_beat = None
    for size in sizes:
        rows = [r for r in table if r['size']==size]
        er = next((r for r in rows if r['method']=='Exact HiGHS'), None)
        if er and er['value'] and er['status'] != 'proven optimal (solver bound meets incumbent)':
            beats = [r for r in rows if r['method']!='Exact HiGHS' and r['value'] and
                     float(r['value']) < float(er['value'])-1e-7]
            if beats:
                first_beat = (size, min(beats, key=lambda r:float(r['value'])), er)
                break
    if first_beat:
        size, winner, er = first_beat
        lines.append(f'First size at which an approximate run beat the unproven HiGHS incumbent: '
                     f'M={size}, {winner["method"]} ({winner["start"]}) cost {winner["value"]} '
                     f'versus exact incumbent {er["value"]}.')
    else:
        lines.append('There is no observed approximate-method improvement over an unproven exact incumbent '
                     'among the available validated runs.')
    for size in sizes:
        for method in METHOD_FILES:
            own = next((r for r in table if r['size']==size and r['method']==method and r['start']=='own'), None)
            rounded = next((r for r in table if r['size']==size and r['method']==method and r['start']=='LP-rounded'), None)
            if own and rounded and own['value'] and rounded['value']:
                diff = float(rounded['value']) - float(own['value'])
                verdict = 'improved' if diff < -1e-7 else 'worsened' if diff > 1e-7 else 'unchanged'
                lines.append(f'LP-rounded effect at M={size}, {method}: {verdict} best cost by '
                             f'{abs(diff):.6f} relative to own start (rounded − own = {diff:+.6f}).')
    lines.append('## Interpretation and limits')
    lines.append('Compare each method to the feasible incumbent and the solver best bound; a solver gap is NOT gapLP. '
                 'An exact method that reaches the 120-second limit without a matching bound has NOT proven optimality. '
                 'Three selected sizes cannot establish a general complexity law or a crossover threshold outside this range; '
                 'additional intermediate sizes and repeated solver timings would test stability of the crossover and solve times.')
    lines.append('Use only source data from results/final_*.csv; the manuscript must not invent other numbers.')
    (outdir/('final_block3_analysis.md' if complete else 'DRAFT_INCOMPLETE_analysis.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke',action='store_true',help='allow short diagnostic runs, never submission-ready')
    ap.add_argument('--sizes',nargs='+',type=int,default=list(FINAL_SIZES))
    ap.add_argument('--seeds',default='0-9')
    ap.add_argument('--budget',type=int,default=100000)
    ap.add_argument('--time-limit',type=float,default=120)
    ap.add_argument('--outdir',default='results')
    ap.add_argument('--methods',nargs='+',choices=['sa','ga'],default=['sa','ga'])
    ap.add_argument('--force',action='store_true',help='rerun saved experiment rows (will clear output CSV files)')
    args=ap.parse_args()
    seeds=parse_seeds(args.seeds)
    sizes=tuple(args.sizes)
    if not args.smoke and (sizes != FINAL_SIZES or seeds != FINAL_SEEDS or args.budget != 100000 or args.time_limit != 120 or set(args.methods)!={'sa','ga'}):
        ap.error('Official full mode must have M=100,200,400; seeds 0..9; 100000 evaluations; 120s; SA and GA. Use --smoke for alternatives.')
    outdir=(ROOT/args.outdir).resolve()
    outdir.mkdir(parents=True,exist_ok=True)
    if not args.smoke and 'smoke' in outdir.parts:
        ap.error('Full runs cannot write into a smoke directory')
    if args.force:
        for key in ('final_scaling_exact.csv','final_scaling_lp.csv','final_scaling_methods.csv'):
            (outdir/key).unlink(missing_ok=True)
    (outdir/'final_environment.json').write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,
                                             scipy=scipy.__version__,numpy=np.__version__,
                                             note='All wall-clock timings depend on hardware; reproducibility is by commands/data/seed.'),indent=2))
    print(f'Experiment: T10 B2 sizes={sizes}, seeds={seeds}, budget={args.budget}, limit={args.time_limit}s, smoke={args.smoke}',flush=True)
    for size in sizes:
        print(f'\n=== Scaled M={size} ===',flush=True)
        d=gen.scaled_instance('B',size,2)
        model=build_model(d)
        path_lp=outdir/'final_scaling_lp.csv'
        lprow=get_row(load_csv(path_lp),size=size)
        start_path=outdir/f'final_LP_rounded_B2_{size}.json'
        if not lprow or not start_path.exists():
            lprow, start=try_lp_round(model,outdir,size)
            add_row(path_lp,LP_COLS,dict(team='T10',story='B',variant=2,size=size,**lprow))
            print('LP:',lprow['status'],'zLP',lprow['z_LP'],'rounded',lprow['rounded_cost'],flush=True)
        else:
            start=json.loads(start_path.read_text())['assign']
            print('LP: existing rounded start loaded',flush=True)
        path_exact=outdir/'final_scaling_exact.csv'
        exact=get_row(load_csv(path_exact),size=size)
        if not exact:
            res=exact_run(model,args.time_limit)
            exact=dict(team='T10',story='B',variant=2,size=size,servers=d['S'],
                       time_limit_s=args.time_limit,**res)
            add_row(path_exact,EXACT_COLS,exact)
            print('EXACT:',res['status'],'obj',res['objective'],'bound',res['best_bound'], 'gap%',res['solver_gap_pct'],flush=True)
        else: print('EXACT: existing row loaded',flush=True)
        for mname in args.methods:
            method='SA transfer' if mname=='sa' else 'GA repair'
            solver=METHOD_FILES[method]
            params=SA_PARAMS if mname=='sa' else GA_PARAMS
            for start_type in ('own','LP-rounded'):
                if start_type=='LP-rounded' and start is None:
                    print(f'WARNING: LP-rounded unavailable for M={size}; not silently substituted.',flush=True)
                    continue
                for seed in seeds:
                    path=outdir/'final_scaling_methods.csv'
                    old=get_row(load_csv(path),size=size,method=method,start=start_type,seed=seed)
                    if old:
                        print(f'skip M={size} {method} {start_type} seed={seed} (already recorded)',flush=True)
                        continue
                    t0=time.perf_counter()
                    try:
                        result=solver(d,2,seed,args.budget,params=params,
                                      start=(list(start) if start_type=='LP-rounded' else None))
                        seconds=time.perf_counter()-t0
                        checker_cost,issues=gen.evaluate_B(d,result['solution'])
                        feasible=not issues
                        if int(result['evals'])!=args.budget: raise AssertionError('evaluation budget not respected')
                        if not np.isclose(checker_cost,float(result['cost']),rtol=1e-7,atol=1e-6):
                            raise AssertionError('returned cost != official checker')
                        status='ok' if feasible else 'infeasible (official checker)'
                        error='; '.join(issues[:5])
                        cost=checker_cost
                        n_evals=result['evals']
                    except Exception as exc:
                        seconds=time.perf_counter()-t0
                        status='error'; error=str(exc); cost=''; feasible=False; n_evals=''
                        print(traceback.format_exc(),file=sys.stderr,flush=True)
                    row=dict(team='T10',story='B',variant=2,size=size,method=method,start=start_type,
                             seed=seed,budget=args.budget,params_json=json.dumps(params,sort_keys=True),
                             cost=cost,feasible=feasible,evals=n_evals,seconds=seconds,status=status,error=error)
                    add_row(path,METHOD_COLS,row)
                    print(f'M={size} {method} {start_type} seed={seed} -> {status}: {fmt(cost)} ({seconds:.2f}s)',flush=True)
    manifest=tabulate(outdir,sizes,seeds,args.budget,args.time_limit,args.smoke)
    print('\n=== COMPLETE ===' if manifest['complete'] else '\n=== INCOMPLETE / DIAGNOSTIC: DO NOT SUBMIT ===',flush=True)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
