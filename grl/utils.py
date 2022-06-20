from pprint import pformat

import numpy as np

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)
