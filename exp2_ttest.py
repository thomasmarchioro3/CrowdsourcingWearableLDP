"""
Experiment 2: Measure agreement rate in independent t test results.

Given two populations, the t-test can give two possible outcomes for
a given confidence value 'alpha':
1) p-value < alpha: the two populations are significantly different
2) p-value > alpha: the two populations are not significantly different

Given a t-test result on the original data, we would like the test on
noisy data to yield the same result.

Procedure:
- For n_iters times:
    - Divide the users into two disjoint random groups
    - Run the t-test between the two groups on the original data
    - Apply LDP to the data to get the noisy data
    - Run the t-test between the two groups on the noisy data
    - Store the outcome

Possible outcomes of the comparison:
- TP: both tests yield p-value < alpha
- FN: test yields p-value < alpha on original data but not on noisy data
- FP: test yields p-value > alpha on original data but not on noisy data
- TN: both tests yield p-value > alpha

When both tests agree on p-value < alpha, one needs to check also the agreement
for the value of t.
"""

if __name__ == '__main__':

    import os
    import pandas as pd
    from numpy.random import seed
    from utils.preprocess import clip
    from utils.exp_utils import montecarlo_ttest_ind, montecarlo_ttest_rel
    from time import time

    seed(42)  # set PRNG seed for reproducibility

    attributes = ['steps', 'calories', 'distance']
    mechanisms = ['laplace', 'piecewise_sub']  # list of mechanisms to be tested
    test_type = 'ind'
    eps_list = [1, 2, 4, 8, 16, 32, 64]  # list of eps values to be tried
    alpha = .05  # threshold for p-value
    n_iters = 1000
    round_ = 0

    montecarlo_ttest = montecarlo_ttest_ind if test_type == 'ind' else montecarlo_ttest_rel

    filename = os.path.join('data', f'lifesnaps_round{round_:d}.csv')
    df = pd.read_csv(filename)  # import dataset table
    df = df.fillna(0)
    exp2_ = 'a' if test_type == 'ind' else 'b'
    resfilename = os.path.join('results', f'experiment2{exp2_}_{n_iters:d}_r{round_:d}.csv')

    print(f"Starting experiment 2 ({test_type})")
    tic = time()

    id_dict = {idx: uid for uid, idx in enumerate(df['id'].unique())}
    df['uid'] = df['id'].apply(lambda x: id_dict[x])
    df = clip(df, attributes)

    n_users_list = list(range(1, len(id_dict)+1, 1))
    if test_type == 'ind':
        n_users_list = list(range(2, len(id_dict)+1, 2))

    reslist = []
    for attr in attributes:
        for mechanism in mechanisms:
            for eps in eps_list:
                for n_users in n_users_list:
                    res = {'attribute': attr, 'mechanism': mechanism, 'eps': eps, 'n_users': n_users}
                    res.update(montecarlo_ttest(df, attr, mechanism, eps, alpha, n_iters, n_users))
                    print(", ".join([f"{key}: {value}" for key, value in res.items()]))
                    reslist.append(res)
    res_df = pd.DataFrame(reslist)

    toc = time()
    print(f"Elapsed time: {toc-tic:.3f} seconds")

    res_df.to_csv(resfilename, index=False)
    # print(res_df)
