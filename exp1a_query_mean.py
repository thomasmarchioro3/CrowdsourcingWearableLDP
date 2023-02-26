"""
Experiment 1: Measure error on the estimate of the average.

Procedure:
- Compute the true average 'mu' for each attribute (steps, calories, etc.) on the original data 'df'
- For n_iters times:
    - Apply LDP on the data with the chosen mechanism, obtaining the noisy data 'dfn'
    - Compute the estimate of the average 'mu_hat' for each attribute on the noisy data
    - Compute the RMSE between 'mu' and 'mu_hat'
- Average the results
"""

if __name__ == '__main__':
    import os
    import pandas as pd
    from numpy.random import seed
    from utils.exp_utils import montecarlo_query
    from utils.preprocess import clip
    from time import time

    seed(42)  # set PRNG seed for reproducibility

    attributes = ['steps', 'calories', 'distance']
    mechanisms = ['laplace', 'piecewise_sub']  # list of mechanisms to be tested
    eps_list = [1, 2, 4, 8, 16, 32, 64]  # list of eps values to be tried

    n_iters = 100
    stats = ['mean', 'rmse', 'nrmse']
    query = 'mean'

    round_ = 0

    filename = os.path.join('data', f'lifesnaps_round{round_:d}.csv')
    df = pd.read_csv(filename)  # import dataset table
    df = df.fillna(0)
    df = clip(df, attributes)

    id_dict = {idx: uid for uid, idx in enumerate(df['id'].unique())}
    df['uid'] = df['id'].apply(lambda x: id_dict[x])
    n_users_list = list(range(1, len(id_dict)+1))
    # n_users_list = list(range(1, 3))

    resfilename = os.path.join('results', f'experiment1a_{query}_{n_iters:d}_r{round_:d}.csv')

    print("Starting experiment 1")
    tic = time()

    reslist = []
    for attr in attributes:
        for mechanism in mechanisms:
            for eps in eps_list:
                for n_users in n_users_list:
                    res = {'attribute': attr, 'mechanism': mechanism, 'eps': eps, 'n_users': n_users}
                    res.update(montecarlo_query(df, attr, mechanism, eps, stats, n_iters, n_users, query=query))
                    print(", ".join([f"{key}: {value}" for key, value in res.items()]))
                    reslist.append(res)
    res_df = pd.DataFrame(reslist)

    res_df.to_csv(resfilename, index=False)
    toc = time()
    print(f"Elapsed time: {toc-tic:.3f} seconds")

