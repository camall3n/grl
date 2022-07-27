## Setup
```bash
pip install -r requirements.txt
```

## Usage

Default behavior is to display the analytical MDP, MC*, and TD solutions:
```bash
python -m grl.run --spec example_7
```

For this example, there is a discrepancy between MC* and TD. To run it with a previously defined memory function:
```bash
python -m grl.run --spec example_7 --use_memory 5
```

If the memory function doesn't resolve it, try to find one that does:
```bash
python -m grl.run --spec example_7 --use_memory 0 --use_grad m
```

```
usage: run.py [-h] [--spec SPEC] [--method METHOD]
              [--n_random_policies N_RANDOM_POLICIES]
              [--use_memory USE_MEMORY] [--use_grad USE_GRAD] [--heatmap]
              [--n_episodes N_EPISODES] [--log] [--seed SEED]
              [-f FOOL_IPYTHON]

optional arguments:
  -h, --help            show this help message and exit
  --spec SPEC           name of POMDP spec; evals Pi_phi policies by default
  --method METHOD       "a"-analytical, "s"-sampling, "b"-both
  --n_random_policies N_RANDOM_POLICIES
                        number of random policies to eval; if set (>0),
                        overrides Pi_phi
  --use_memory USE_MEMORY
                        use memory function during policy eval if set
  --use_grad USE_GRAD   find policy ("p") or memory ("m") that minimizes any
                        discrepancies by following gradient
  --heatmap             generate a policy-discrepancy heatmap for the given
                        POMDP
  --n_episodes N_EPISODES
                        number of rollouts to run
  --log                 save output to logs/
  --seed SEED           seed for random number generators
  -f FOOL_IPYTHON, --fool-ipython FOOL_IPYTHON
```

## Run tests
```bash
pip install pytest
pytest tests
```

## Formatting
```bash
pip install yapf
yapf --recursive --in-place .
```
