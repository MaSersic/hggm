import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import matlab.engine
import os

import helpers

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


def init_matlab():
    print("Matlab init started")
    eng = matlab.engine.start_matlab()
    print("Matlab started")

    cwd = os.getcwd()
    "HGGM_Code"
    s = eng.genpath(".\\HGGM_Code\\Source code and test scripts")
    eng.addpath(s, nargout=0)
    return eng


def hggm(time_series, lam, I_n, I_in, I_p, I_g, I_B, eng, lag=None):
    if lag is None:
        max_lag = 20
        lag = int(helpers.find_lag_aic(max_lag, time_series))

    dist = helpers.find_dist(len(time_series[0])-1, I_n, I_in, I_p, I_g, I_B)
    coeffs = adaptive_lasso_regression(time_series, lag, lam, I_n, I_in, I_p, I_g, I_B, eng)

    betas = []
    for row in coeffs:
        row = np.array(row)
        b = []
        for coefs in row:
            b.append(np.array(coefs).max())
        betas.append(b[0:len(coeffs)])

    return np.array(betas), (dist, lag)


def hggm_matlab(series, m_lag, lam, I_n, I_p, I_g, I_B, eng):
    """
    Wrapper for the
    """

    lag = int(helpers.find_lag_aic(m_lag, series))
    print('HGGM is running ...... ')
    print("Lag:")
    print(lag)
    time_series = series.copy().astype(float).T

    # Adjust indexing for matlab
    I_n = I_n + 1.0
    I_p = I_p + 1.0
    I_g = I_g + 1.0
    I_B = I_B + 1.0

    AD_coefs, runtime = eng.lasso_regression(time_series, lag, lam, I_n, I_p, I_g, I_B, nargout=2)

    betas = []
    for row in AD_coefs:
        row = np.array(row)
        b = []
        for coefs in row:
            b.append(np.array(coefs).max())
        betas.append(b[0:len(AD_coefs)])

    print('HGGM is done ...... ')

    return np.array(betas)


def adaptive_lasso_regression(series, lag, lam, I_n, I_in, I_p, I_g, I_B, eng):
    series = series.T

    p = len(series[:, 0])
    T = len(series[0])
    coeffs = []

    D = p * lag
    N = T - lag

    PHI = np.zeros((N, D))

    t = np.zeros(N)
    for i in range(0, N):
        for j in range(0, p):
            cur_row_start = j * lag
            cur_row_end = cur_row_start + lag
            PHI[i, cur_row_start: cur_row_end] = series[j, i: (i + lag)]

    # do regression with each time series as target
    for target_row in range(0, p):
        t[0: N] = series[target_row, lag: T]

        # gaussian
        if target_row in I_n:
            coef = eng.lasso_reg(PHI, t, lag, lam, True, False, False, False, False, nargout=1)

        # inverse gaussian
        if target_row in I_in:
            coef = eng.lasso_reg(PHI, t, lag, lam, False, True, False, False, False, nargout=1)

        # poisson
        if target_row in I_p:
            coef = eng.lasso_reg(PHI, t, lag, lam, False, False, True, False, False, nargout=1)

        # gamma
        if target_row in I_g:
            coef = eng.lasso_reg(PHI, t, lag, lam, False, False, False, True, False, nargout=1)

        # binomial
        if target_row in I_B:
            coef = eng.lasso_reg(PHI, t, lag, lam, False, False, False, False, True, nargout=1)

        coeffs.append(coef)

    return coeffs


def find_lag_supervised(max_lag, data, truth, max_lam, threshold, eng):
    lags = range(1, int(max_lag))
    I_p = np.array([]).astype(float)  # poisson
    I_b = np.array([]).astype(float)  # binomial
    I_n, I_in, I_g = helpers.fit_data(data.astype(float))

    fmeasures = []
    for lag in lags:
        avg_fm = 0.0
        for i in range(0,5):
            coefs_hggm = hggm(data, max_lam, I_n, I_in, I_p, I_g, I_b, eng, lag)[0]
            result = helpers.classify_coefs(coefs_hggm, threshold, len(data[0]))

            fm = helpers.fmeasure(truth, result)
            avg_fm = avg_fm + fm
        fmeasures.append([avg_fm / 5, lag])

    fmeasures = np.array(fmeasures)
    fmeasures = fmeasures[fmeasures[:,0].argsort()[::-1]]
    lag = fmeasures[0][1]

    print(fmeasures[0][0])
    return int(lag)