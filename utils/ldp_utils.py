import numpy as np
import pandas as pd

from .preprocess import scale, unscale
from .ldp.laplace_mechanism import apply_laplace
from .ldp.piecewise_mechanism_sub import apply_pm_sub


def apply_ldp_df(df, eps, num_attributes, mechanism='laplace'):

    if isinstance(mechanism, str):
        if mechanism == 'laplace':
            mechanism = apply_laplace
        elif mechanism == 'piecewise_sub':
            mechanism = apply_pm_sub
        else:
            raise ValueError("Unknown mechanism")

    m = len(num_attributes)

    dfn_scaled = scale(df, num_attributes)
    for attr in num_attributes:
        dfn_scaled[attr] = dfn_scaled[attr].apply(lambda x: mechanism(x, eps=eps/m))  # divide the privacy budget among the m attributes

    dfn = unscale(dfn_scaled, num_attributes)

    return dfn


