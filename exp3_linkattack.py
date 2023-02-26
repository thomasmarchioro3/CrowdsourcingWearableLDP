"""
Experiment 3: Measure accuracy of linking attack
"""

if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from utils.exp_utils import montecarlo_linking
    from utils.preprocess import clip
    from numpy.random import seed
    from time import time
    import argparse

    seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=str, default='steps_calories')
    args = vars(parser.parse_args())

    attributes = args["attributes"].split("_")
    # attributes = ['steps', 'calories']
    # attributes = ['steps']
    mechanisms = ['laplace', 'piecewise_sub']
    eps_list = [1, 2, 4, 8, 16, 32, 64, np.inf]

    n_iters = 100
    round_ = 0  # written with _ to avoid shadowing built-in function

    attr_str = ''.join([a[0].upper() for a in attributes])  # construct the string with the first letter of each attr
    # print(attr_str)
    # exit()

    filename = os.path.join('data', f'lifesnaps_round{round_:d}.csv')
    df = pd.read_csv(filename)
    df = df.fillna(0)
    df = clip(df, attributes)

    print(f"Starting experiment 3 on Round {round_:d}")
    tic = time()

    id_dict = {idx: uid for uid, idx in enumerate(df['id'].unique())}
    df['uid'] = df['id'].apply(lambda x: id_dict[x])
    # df = df.drop(columns='id')
    n_users_list = list(range(1, len(id_dict)+1))

    resfilename = os.path.join('results', f'experiment3_{attr_str}_{n_iters:d}_r{round_:d}.csv')

    reslist = []
    for mechanism in mechanisms:
        for eps in eps_list:
            for n_users in n_users_list:
                res = {'mechanism': mechanism, 'eps': eps, 'n_users': n_users}
                res.update(montecarlo_linking(df, attributes, mechanism, eps, n_iters, n_users))
                print(", ".join([f"{key}: {value}" for key, value in res.items()]))
                reslist.append(res)

    res_df = pd.DataFrame(reslist)

    toc = time()
    print(f"Elapsed time: {toc-tic:.3f} seconds")
    res_df.to_csv(resfilename, index=False)
