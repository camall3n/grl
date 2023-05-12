import numpy as np
import scipy
from grl.utils.math import arg_hardmax, arg_boltzman, mellowmax, arg_mellowmax, one_hot

x = np.repeat(np.arange(10)[None, :], 2, axis=0)

assert arg_hardmax(x).shape == x.shape
assert arg_boltzman(x, beta=1.).shape == x.shape
assert arg_mellowmax(x, beta=3.9).shape == x.shape

assert mellowmax(x, beta=20).shape == x.shape[:1]

x = np.repeat(np.arange(10)[None, :], 2, axis=0).T

assert arg_hardmax(x).shape == x.shape
assert arg_boltzman(x, beta=1).shape == x.shape
assert arg_mellowmax(x, beta=3.9).shape == x.shape

#%%
shape = (2,3,4,5)
x = np.reshape(np.arange(np.prod(shape)), shape)

for axis in range(-2, 4):
    assert np.allclose(1, arg_mellowmax(x, axis=axis).sum(axis))

for axis in range(-2, 4):
    assert np.allclose(1, arg_hardmax(x, axis=axis).sum(axis))

for axis in range(-2, 4):
    assert np.allclose(1, arg_boltzman(x, axis=axis).sum(axis))
