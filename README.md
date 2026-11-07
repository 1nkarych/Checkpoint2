
# Checkpoint 4 - Team T10

## Instance
- Storyline: B (Cloud)
- Variant: B2
- Instance seed: 10
- Instance call: `cp34_instance(10, "B", 2)`

## SA-I: Hafizi Nazifullah
- Role: Simulated Annealing Implementer
- Configuration: SA, transfer
- Partner: Shaken Daniyal (SA-T)

## Implementation
- `src/meta/sa.py`: Simulated Annealing implementation
- `tests/test_sa_i.py`: SA-I tests
- `experiments/sa_i_analysis.py`: Acceptance rate analysis
- `experiments/sa_i_worst_seed.py`: Improving-move and acceptance analysis

The algorithm uses transfer moves and checks CPU, memory, and zone restrictions. Only feasible moves are accepted.

## Testing

Run the SA-I tests:

```bash
python -m unittest discover -s tests -p "test_sa_i.py" -v
```

## AI Assistance Declaration

| Tool | Used for | Files | Independently checked |
|---|---|---|---|
| OpenAI ChatGPT | Implementation guidance, code drafting, debugging, and test preparation | SA-I Python files | Executed five tests, checked feasibility and evaluation counts against the generator |
