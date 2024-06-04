import scipy.stats
import numpy as np
from statsmodels.tsa.api import VAR

def fit_data(data, distributions={
                            'gaussian': scipy.stats.norm,
                            'gamma': scipy.stats.gamma,
                            'inverse_gaussian': scipy.stats.invgauss
                        }):

    I_n = np.array([]).astype(float)  # normal
    I_in = np.array([]).astype(float)  # inverse normal
    I_g = np.array([]).astype(float)  # gamma

    for index, row in enumerate(data.transpose()):
        result = {i: None for i in distributions}
        for i, cd in distributions.items():
            fit = cd.fit(row)
            result[i] = scipy.stats.kstest(row, cd.cdf, args=fit)

        dic = {key: value for key, value in sorted(result.items(), key=lambda item: item[1][1], reverse=True)}
        best = list(dic.items())[0][0]
        match best:
            case 'gaussian':
                I_n = np.concatenate([I_n, np.array([index]).astype(float)])
            case 'gamma':
                # All values in gamma and inverse Gaussian distributions must be positive
                if np.any(row <= 0.0):
                    I_n = np.concatenate([I_n, np.array([index]).astype(float)])
                else:
                    I_g = np.concatenate([I_g, np.array([index]).astype(float)])
            case 'inverse_gaussian':
                # All values in gamma and inverse Gaussian distributions must be positive
                if np.any(row <= 0.0):
                    I_n = np.concatenate([I_n, np.array([index]).astype(float)])
                else:
                    I_in = np.concatenate([I_in, np.array([index]).astype(float)])

    return np.sort(I_n), np.sort(I_in), np.sort(I_g)


def find_lag_aic(max_lag, data):
    lags = range(1, int(max_lag))
    var_model = VAR(data)

    """
    VAR select order can throw an error if given a too large maxlags parameter. If max_lag is
    larger than the default chosen by select order fall back to the default.
    """

    if max_lag is not None and max_lag > 12 * (len(data) / 100) ** (1 / 4):
        max_lag = None

    lag = var_model.select_order(max_lag).aic

    if lag == 0:
        return 1
    return lag


def scale_betas(betas):
    sum_b_max = betas.sum()
    for i in range(0,len(betas)):
        betas[i] = betas[i]/sum_b_max
    return betas


def scale_data(data):
    return (data - np.min(data)) / (np.max(data))


def find_dist(index, I_n, I_in, I_p, I_g, I_B):
    if index in I_n: return "Gaussian"
    if index in I_in: return "Inverse gaussian"
    if index in I_p: return "Poisson"
    if index in I_g: return "Gamma"
    if index in I_B: return "Binomial"
    return "Not found"


def classify_coefs(coefs, threshold, p):
    m = -1
    for row in coefs:
        if max(row) > m:
            m = max(row)

    matrix = np.zeros((len(coefs), len(coefs)))
    for i, row in enumerate(coefs):
        size = len(row)
        if size > p:
            l = size / p
            for j in range(0, int(p)):
                if row[int(j*l)][0] >= threshold:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0

        else:
            for j, field in enumerate(row):
                if field >= threshold:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
    return matrix.astype(int)


def fmeasure(true, est):
    N = est.shape[0]
    M = est.shape[1]
    same = 0

    for i in range(0,N):
        for j in range(0,M):
            if est[i][j] == 1 and true[i][j] == 1:
                same = same + 1
    # precision

    truth_nnz = len(np.nonzero(true)[0])
    est_nnz = len(np.nonzero(est)[0])
    if est_nnz == 0:
        p = 0
    else:
        p = same / est_nnz

    # recall

    r = same / truth_nnz

    # F - measure

    if p == 0 and r == 0:
        f_measure = 0
    else:
        f_measure = (2 * p * r) / (p + r)

    return f_measure
