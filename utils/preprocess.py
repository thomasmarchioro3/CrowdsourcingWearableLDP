import numpy as np

# not interested in values above the max or below the min
MIN_DICT = {
    'steps': 0,
    'calories': 0,
    'distance': 0,
    'scaled': -1,
}

MAX_DICT = {
    'steps': 20000,
    'calories': 6000,
    'distance': 15000,
    'scaled': 1,
}


def scale(df, attributes):
    df = df.copy()
    for attr in attributes:
        df[attr] = np.clip((df[attr] - MIN_DICT[attr])/(MAX_DICT[attr]-MIN_DICT[attr]), 0, 1)  # scale/clip to range [0,1]
        df[attr] = df[attr]*(MAX_DICT['scaled']-MIN_DICT['scaled'])+MIN_DICT['scaled']
    return df


def unscale(df, attributes):
    df = df.copy()
    for attr in attributes:
        df[attr] = (df[attr] - MIN_DICT['scaled'])/(MAX_DICT['scaled']-MIN_DICT['scaled'])  # scale/clip to range [0,1]
        df[attr] = df[attr]*(MAX_DICT[attr]-MIN_DICT[attr])+MIN_DICT[attr]
    return df


def clip(df, attributes):
    df = df.copy()
    for attr in attributes:
        df[attr] = np.clip(df[attr], MIN_DICT[attr], MAX_DICT[attr])  # scale/clip to range [0,1]
    return df
