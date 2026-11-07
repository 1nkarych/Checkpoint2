#!/usr/bin/env bash
# Reproduces every number, table and figure of Telgarin_CP4.pdf (team T10, cp34_instance(10,'B',2)).
set -e
python check_env.py --instance-seed 10 --story B --variant 2
python -m unittest tests.test_ga_a -v
COMMON="--instance-seed 10 --story B --variant 2 --team T10 --role GA-A --budget 100000"
# Table 1 / F1: final runs, seeds 0-9, rule parameters (N=30, pc=0.9, 1-2 mutated components, tournament 2, elitism 1)
python experiments/run_final.py --module src.meta.ga $COMMON --configuration "GA, repair" --seeds 0-9 --out results/Telgarin_runs.csv
# random-search baseline, same budget and seeds
python experiments/run_final.py --module src.meta.random_search $COMMON --configuration random --seeds 0-9 --out results/Telgarin_random.csv
# required item of GA-A: effect of the population size N (N=30 is the rule value, also in Telgarin_runs.csv)
for N in 10 30 100; do
  python experiments/run_final.py --module src.meta.ga $COMMON --configuration "GA, repair, N=$N" --seeds 0-9 --params "{\"N\": $N}" --out results/Telgarin_Nexp.csv --append
done
# traces, diversity per generation, worst-seed checks (Table 2), table and figures
python experiments/analysis.py
python experiments/make_report.py
