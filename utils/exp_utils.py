import numpy as np
from numpy import sqrt
from .ldp_utils import apply_ldp_df
from scipy.stats import ttest_ind, ttest_rel
from .preprocess import scale, MIN_DICT, MAX_DICT


# needed for exp1
def compute_stats(q, q_hat, stats=None):
    if stats is None:
        stats = ['rmse', 'nrmse']
    res = dict()
    if 'mean' in stats:
        res['mean'] = (q_hat - q).mean()
    rmse = sqrt(((q_hat - q)**2).mean())
    # print(rmse)
    if 'rmse' in stats:
        res['rmse'] = rmse
    if 'nrmse' in stats:
        nrmse = rmse/(q.max()-q.min())
        res['nrmse'] = nrmse
    return res


def compute_aggr_per_day(df, attr, aggr='mean'):
    ts_aggr = df[['date']+[attr]].groupby(['date'])
    if aggr == 'mean':
        x = ts_aggr.mean()
    elif aggr == 'sum':
        x = ts_aggr.sum()
    elif aggr == 'std':
        x = ts_aggr.std()
    elif aggr == 'var':
        x = ts_aggr.var()
    else:
        x = ts_aggr.mean()
    return x.to_numpy()


def compute_icdf_laplace_sample(xu, threshold, sensitivity, eps):
    p = .5*np.exp(-eps*abs(xu-threshold)/sensitivity)
    return p if xu <= threshold else 1-p


def compute_icdf_per_day(df, attr, mechanism, eps=np.inf, threshold=10000):
    df = df.copy()
    df['p_over'] = df['steps'].apply(lambda xu: 1 if xu > threshold else 0)
    if mechanism == 'laplace':
        sensitivity = MAX_DICT[attr] - MIN_DICT[attr]
        df['p_over'] = df['steps'].apply(lambda xu: compute_icdf_laplace_sample(xu, threshold, sensitivity, eps))
    return compute_aggr_per_day(df, 'p_over', 'sum')


# def estimate_std_noisy(dfn, eps, attr, mechanism):
#     var_xn = compute_aggr_per_day(dfn, attr, aggr='var')
#     if mechanism == 'laplace':
#         sens = MAX_DICT[attr] - MIN_DICT[attr]
#         var_e = 2*((sens/eps)**2)
#         var_hat = var_xn - var_e
#     elif mechanism == 'piecewise_sub':
#         mu_hat = compute_aggr_per_day(dfn, attr, aggr='mean')
#         expeps = np.exp(eps)
#         k1 = (expeps+1)/(expeps-1)
#         k2 = (2*expeps*((expeps+1)**3 + expeps-1))/(3*(expeps**2)*(expeps-1)**2)
#         var_hat = (var_xn - k2)/k1 #- mu_hat**2
#     else:
#         var_hat = var_xn
#     return np.sqrt(np.maximum(0, var_hat))


def montecarlo_query(df, attr, mechanism, eps, stats, n_iters, n_users, query='mean', threhold=10000):
    stats_coll = {stat: [] for stat in stats}
    for n in range(n_iters):
        # sample n_users at random
        uidx = np.random.choice(df['uid'].unique(), size=n_users, replace=False)
        df_ = df[df['uid'].isin(uidx)]
        # apply ldp
        dfn = apply_ldp_df(df_, eps, [attr], mechanism=mechanism)
        q = [np.nan]
        q_hat = [np.nan]
        if query == 'mean':
            q = compute_aggr_per_day(df_, attr=attr, aggr=query)
            q_hat = compute_aggr_per_day(dfn, attr=attr, aggr=query)
        # if query == 'std':
        #     q_hat = estimate_std_noisy(dfn, eps, attr, mechanism)
        if query == 'count':
            q = compute_icdf_per_day(df_, attr, mechanism='none')
            q_hat = compute_icdf_per_day(dfn, attr, mechanism, eps, threhold)#.round()
            # print(np.transpose([q, q_hat]))
        res = compute_stats(q, q_hat, stats=stats)
        for stat in stats:
            stats_coll[stat].append(res[stat])
    return {stat: np.mean(stats_coll[stat]) for stat in stats}
#
# def montecarlo_stats(df, attr, mechanism, eps, stats, n_iters, query='mean', return_stat=None):
#     stats_coll = {stat: [] for stat in stats}
#     for n in range(n_iters):
#         dfn = apply_ldp_df(df, eps, [attr], mechanism=mechanism)
#         q = compute_aggr_per_day(df, attr=attr, aggr=query)
#         q_hat = q.copy()
#         if query == 'mean':
#             q_hat = compute_aggr_per_day(dfn, attr=attr, aggr='mean')
#         elif query == 'std':
#             q_hat = estimate_std_noisy(dfn, eps, attr, mechanism)
#         res = compute_stats(q, q_hat, stats=stats)
#         for stat in stats:
#             stats_coll[stat].append(res[stat])
#     if return_stat == 'mean':
#         return {stat: np.mean(stats_coll[stat]) for stat in stats}
#     return {stat: stats_coll[stat] for stat in stats}


# needed for exp2
def montecarlo_ttest_ind(df, attr, mechanism, eps, alpha, n_iters, n_users):

    tot_users = len(df['uid'].unique())
    n_ = n_users // 2  # number of users per group

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n_p = 0  # counter for actual positives of t-test (ground truth)

    t_agreement = []
    for i in range(n_iters):
        ground_truth = False  # whether t-test on the actual data has p<alpha (default is False)

        # sample n_users participants
        idx = np.random.permutation(tot_users)
        idx1 = idx[:n_]
        idx2 = idx[n_:(2*n_)]

        dfx1 = df[df['uid'].isin(idx1)]
        dfx2 = df[df['uid'].isin(idx2)]

        # compute true means on the original data
        mu1 = compute_aggr_per_day(dfx1, attr=attr)
        mu2 = compute_aggr_per_day(dfx2, attr=attr)

        t, p = ttest_ind(mu1, mu2)
        if p < alpha:
            ground_truth = True
            n_p += 1

        # apply LDP
        dfn1 = apply_ldp_df(dfx1, eps, [attr], mechanism=mechanism)
        dfn2 = apply_ldp_df(dfx2, eps, [attr], mechanism=mechanism)

        mun1 = compute_aggr_per_day(dfn1, attr=attr)
        mun2 = compute_aggr_per_day(dfn2, attr=attr)

        t_, p_ = ttest_ind(mun1, mun2)

        if ground_truth:
            if p_ < alpha:
                tp += 1
                t_agreement.append(1 if t*t_ > 0 else 0)
            else:
                fn += 1
        else:
            if p_ > alpha:
                tn += 1
            else:
                fp += 1

    rate_p = n_p/n_iters

    accuracy = (tp + tn)/(tp + tn + fp + fn)

    if tp + fp > 0:
        precision = tp/(tp + fp)
    else:
        precision = np.nan

    if tp + fn > 0:
        recall = tp/(tp + fn)
    else:
        recall = np.nan

    if tn + fp > 0:
        specificity = tn/(tn + fp)
    else:
        specificity = np.nan

    if (tp + fn > 0) and (tn + fp > 0):
        balanced_accuracy = rate_p*recall + (1-rate_p)*specificity
    else:
        balanced_accuracy = np.nan

    if tp > 0:
        t_agree_rate = np.mean(t_agreement)
    else:
        t_agree_rate = np.nan

    res = {
        'p_rate': rate_p,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        't_agree_rate': t_agree_rate
        }
    return res


def montecarlo_ttest_rel(df, attr, mechanism, eps, alpha, n_iters, n_users):

    tot_users = len(df['uid'].unique())

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n_p = 0  # counter for actual positives of t-test (ground truth)

    t_agreement = []
    for i in range(n_iters):
        ground_truth = False  # whether t-test on the actual data has p<alpha (default is False)

        # sample n_users participants
        idx = np.random.permutation(tot_users)[:n_users]

        dfx = df[df['uid'].isin(idx)]

        # compute true means on the original data
        mu = compute_aggr_per_day(dfx, attr=attr)
        timestamps = len(mu)
        t_half = timestamps // 2  # number of users per group

        mu1 = mu[:t_half]
        mu2 = mu[t_half:(2*t_half)]

        # print(mu1.mean(), mu2.mean())
        # exit()

        t, p = ttest_rel(mu1, mu2)
        if p < alpha:
            ground_truth = True
            n_p += 1

        # apply LDP
        dfn = apply_ldp_df(dfx, eps, [attr], mechanism=mechanism)

        mun = compute_aggr_per_day(dfn, attr=attr)
        mun1 = mun[:t_half]
        mun2 = mun[t_half:(2*t_half)]

        t_, p_ = ttest_rel(mun1, mun2)

        if ground_truth:
            if p_ < alpha:
                tp += 1
                t_agreement.append(1 if t*t_ > 0 else 0)
            else:
                fn += 1
        else:
            if p_ > alpha:
                tn += 1
            else:
                fp += 1

    rate_p = n_p/n_iters

    accuracy = (tp + tn)/(tp + tn + fp + fn)

    if tp + fp > 0:
        precision = tp/(tp + fp)
    else:
        precision = np.nan

    if tp + fn > 0:
        recall = tp/(tp + fn)
    else:
        recall = np.nan

    if tn + fp > 0:
        specificity = tn/(tn + fp)
    else:
        specificity = np.nan

    if (tp + fn > 0) and (tn + fp > 0):
        balanced_accuracy = rate_p*recall + (1-rate_p)*specificity
    else:
        balanced_accuracy = np.nan

    if tp > 0:
        t_agree_rate = np.mean(t_agreement)
    else:
        t_agree_rate = np.nan

    res = {
        'p_rate': rate_p,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        't_agree_rate': t_agree_rate
        }
    return res


# needed for exp3
def users_df_to_array(df, attributes):

    if len(attributes) > 1:  # for reports with multiple attributes
        dfs = scale(df, attributes)

    else:
        dfs = df.copy()  # no need to rescale if a single attribute is considered

    xfs = []
    for attr in attributes:  # create a matrix (users, timestamps) per each attributes
        xf = np.asarray(dfs.groupby('uid', sort=True)[attr].apply(list).to_list())
        xfs.append(xf)

    x = np.stack(xfs, axis=-1)  # concatenate the matrices to obtain an array of dims (n_users, timestamps, n_attr)

    return x


# def predict_uid(x_all_t, x_u_all):
#     n_attr = x_all_t.shape[-1]
#     # construct distances matrix of size
#     distances = np.sum([np.subtract.outer(x_all_t[:, a], x_u_all[:, a])**2 for a in range(n_attr)], axis=0)
#     uidx, _ = np.unravel_index(distances.argmin(), distances.shape)  # find the argmin w.r.t. the users
#     return uidx


# def montecarlo_linking(df, attributes, mechanism, eps, n_iters, n_users):
#
#     tot_users, timestamps, _ = users_df_to_array(df, attributes).shape
#
#     n_correct = 0
#     for _ in range(n_iters):
#         uidx = np.random.choice(np.arange(tot_users), size=n_users, replace=False)
#         df_ = df[df['uid'].isin(uidx)]
#         if eps < np.inf:
#             dfn = apply_ldp_df(df_, eps, attributes, mechanism=mechanism)
#         else:
#             dfn = df.copy()
#
#         x = users_df_to_array(df_, attributes)
#         xn = users_df_to_array(dfn, attributes)
#
#         for u in range(n_users):
#             x_u_all = x[u, :, :]
#             for t in range(timestamps):
#                 x_all_t = xn[:, t, :]
#                 if predict_uid(x_all_t, x_u_all) == u:
#                     n_correct += 1
#
#     # print(n_iters*n_users*timestamps)
#     # exit()
#
#     accuracy = n_correct / (n_iters*n_users*timestamps)
#
#     res = {'accuracy': accuracy}
#     return res

def predict_uid(x_all_t, x_u_t):
    n_attr = x_all_t.shape[-1]
    # construct distances matrix of size
    distances = np.sum([np.abs(x_all_t[:, a] - x_u_t[a]) for a in range(n_attr)], axis=0)
    uidx = distances.argmin()  # find the argmin w.r.t. the users
    return uidx


def montecarlo_linking(df, attributes, mechanism, eps, n_iters, n_users):

    tot_users, timestamps, _ = users_df_to_array(df, attributes).shape

    n_correct = 0
    for _ in range(n_iters):
        uidx = np.random.choice(np.arange(tot_users), size=n_users, replace=False)
        df_ = df[df['uid'].isin(uidx)]
        if eps < np.inf:
            dfn = apply_ldp_df(df_, eps, attributes, mechanism=mechanism)
        else:
            dfn = df_.copy()

        x = users_df_to_array(df_, attributes)
        xn = users_df_to_array(dfn, attributes)

        for u in range(n_users):
            x_u_all = x[u, :, :]
            for t in range(timestamps):
                xn_all_t = xn[:, t, :]
                if predict_uid(xn_all_t, x_u_all[t, :]) == u:
                    n_correct += 1

    accuracy = n_correct / (n_iters*n_users*timestamps)

    res = {'accuracy': accuracy}
    return res


def montecarlo_linking_cm(df, attributes, mechanism, eps, n_iters, n_users):

    tot_users, timestamps, _ = users_df_to_array(df, attributes).shape

    cm = np.zeros((tot_users, tot_users))
    for i in range(n_iters):
        uidx = np.random.choice(np.arange(tot_users), size=n_users, replace=False)
        df_ = df[df['uid'].isin(uidx)]
        if eps < np.inf:
            dfn = apply_ldp_df(df_, eps, attributes, mechanism=mechanism)
        else:
            dfn = df_.copy()

        x = users_df_to_array(df_, attributes)
        xn = users_df_to_array(dfn, attributes)

        for u_true in range(n_users):
            x_u_all = x[u_true, :, :]
            for t in range(timestamps):
                xn_all_t = xn[:, t, :]
                u_pred = predict_uid(xn_all_t, x_u_all[t, :])
                cm[uidx[u_true], uidx[u_pred]] += 1

    return cm
