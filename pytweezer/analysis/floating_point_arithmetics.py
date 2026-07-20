import numbers
import sys
import numpy as np


def round_floating_prec(value):
    if not np.isscalar(value):
        for enum, v in enumerate(value):
            value[enum] = round_floating_prec(v)
    else:
        if isinstance(value, numbers.Number) and type(value) is not bool:
            return np.round(value, sys.float_info.dig - 3)
    return value
