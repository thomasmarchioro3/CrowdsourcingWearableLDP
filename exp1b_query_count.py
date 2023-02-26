if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from numpy.random import seed
    from utils.exp_utils import montecarlo_query
    from utils.preprocess import clip
    from time import time

    seed(42)  # set PRNG seed for reproducibility

    threshold_dict = {'steps': 10000, 'calories': 3000, 'distance': 7500}
    attributes = threshold_dict.keys()
    mechanisms = ['laplace']  # list of mechanisms to be tested
    eps_list = [1, 2, 4, 8, 16, 32, 64]  # list of eps values to be tried
    # eps_list = [32, 64]

    n_iters = 1000
    stats = ['mean', 'rmse', 'nrmse']

    round_ = 0

    filename = os.path.join('data', f'lifesnaps_round{round_:d}.csv')
    df = pd.read_csv(filename)  # import dataset table
    df = df.fillna(0)
    df = clip(df, attributes)

    # for attr in attributes:
    #     print(df[attr].median())
    # exit()

    id_dict = {idx: uid for uid, idx in enumerate(df['id'].unique())}
    df['uid'] = df['id'].apply(lambda x: id_dict[x])
    n_users_list = list(range(1, len(id_dict)+1))
    # n_users_list = list(range(40, len(id_dict)+1))

    resfilename = os.path.join('results', f'experiment1b_count_{n_iters:d}_r{round_:d}.csv')

    print("Starting experiment 1b")
    tic = time()

    reslist = []
    for attr, threshold in threshold_dict.items():
        for mechanism in mechanisms:
            for eps in eps_list:
                for n_users in n_users_list:
                    res = {'attribute': attr, 'mechanism': mechanism, 'eps': eps, 'n_users': n_users}
                    res.update(montecarlo_query(df, attr, mechanism, eps, stats, n_iters, n_users, query='count', threhold=threshold))
                    print(", ".join([f"{key}: {value}" for key, value in res.items()]))
                    reslist.append(res)
    res_df = pd.DataFrame(reslist)

    res_df.to_csv(resfilename, index=False)
    toc = time()
    print(f"Elapsed time: {toc-tic:.3f} seconds")

