## Setup
```bash
pip install -r requirements.txt
```

## Usage
```
usage: run.py [-h] [--spec SPEC] [--no_gamma]
              [--n_random_policies N_RANDOM_POLICIES] [--use_grad]
              [--n_steps N_STEPS] [--max_rollout_steps MAX_ROLLOUT_STEPS]
              [--log] [-f FOOL_IPYTHON] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --spec SPEC           Name of POMDP spec
  --no_gamma            do not discount the occupancy expectation in policy
                        eval
  --n_random_policies N_RANDOM_POLICIES
                        number of random policies to run--if not set, then use
                        specified Pi_phi instead
  --use_grad            find policy that minimizes any discrepancies by
                        following gradient
  --n_steps N_STEPS     number of rollouts to run
  --max_rollout_steps MAX_ROLLOUT_STEPS
                        max steps for mc rollouts
  --log                 save output to logs/
  -f FOOL_IPYTHON, --fool-ipython FOOL_IPYTHON
  --seed SEED
```

Example of commonly used command:
```bash
python -m grl.run --spec example_3 --no_gamma --log
```

## Run tests
```bash
pip install pytest
pytest tests
```

## Formatting
```bash
yapf --recursive --in-place .
```
