# .POMDP files
This directory contains files pertaining to POMDPs as specified by the 
`.POMDP` file extension described [here](http://pomdp.org/code/pomdp-file-spec.html).

## .POMDP file extension
As explained on Tony Cassandra's POMDP website [here](http://pomdp.org/code/pomdp-file-spec.html).

## .alpha file extension
POMDP solver solutions, usually from running the `pomdp-solve` package on a particular `.POMDP` file.

The `pomdp-solve` package is described [here](http://pomdp.org/code/index.html).

Contents and format of the `.alpha` files are described [here](http://pomdp.org/code/alpha-file-spec.html). 
In short, each pair of lines are an action index, followed by `V1, ..., VN`, which are the weights
associated with each belief state for the particular action in a particular portion of the belief state.
To find the best action for a particular belief state, we take the dot product with the belief state across
all weights, and take the action index corresponding to the highest value from the dot product.

## .npy file extension
A file generated from parsing each POMDP's `.POMDP` and `.alpha` files. Each
`.npy` file contains a dictionary consisting of the following for each POMDP and
it's solution:

```python
{
    'p0': np.ndarray, # initial state probabilities
    'actions': np.ndarray, # an array of all action indices as ordered (top to bottom) in the .alpha files
    'coeffs': np.ndarray, # an array of all vector coefficients as ordered (top to bottom) in the .alpha files
    'max_start_idx': int # the index of the highest-value vector given the start state distribution. 
}
```

These files are generated with the script `scripts/parse_alpha.py`.