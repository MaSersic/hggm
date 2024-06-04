from os.path import dirname, abspath, join as pjoin
import numpy as np
import HGGM
import helpers

data_dir = pjoin(dirname(abspath(__file__)), 'HGGM_Code', 'synthetic_data')
matlab_file_path = pjoin(data_dir, 'octave')

eng = HGGM.init_matlab()

# Real test data
truth = np.genfromtxt('Gaussian_Solution.txt', delimiter=",", dtype=int)
series = np.genfromtxt('Gaussian.csv', delimiter=",", dtype=int)
p = 7

I_n = np.array([0,1,2,3,4,5,6]).astype(float)  # normal
I_in = np.array([]).astype(float)
I_p = np.array([]).astype(float)  # poisson
I_g = np.array([]).astype(float)  # gamma
I_b = np.array([]).astype(float)  # binomial

# Calculations

results = []
results_ml = []
lam = 7.0
#for lam in range(1, 100):
#    lam = float(lam)
for threshold in np.arange(0.0, 0.1, 0.02):
    lag = HGGM.find_lag_supervised(5.0, series, truth, lam, threshold, eng)
    avg_py = 0.0
    for i in range(0, 5):
        coefs_py = HGGM.hggm(series, lam, I_n, I_in, I_p, I_g, I_b, eng, lag)[0]
        result_py = helpers.classify_coefs(coefs_py, threshold, p)
        fmeasure_py = helpers.fmeasure(truth, result_py)
        avg_py = avg_py + fmeasure_py

        results.append([avg_py/5, lag, threshold, lam])
results = np.array(results)
results = results[results[:,0].argsort()[::-1]]

print("Best fmeasure:")
print(results[0][0])
print("Lag, threshold")
print(results[0][1])
print(results[0][2])

lag = results[0][1]
threshold = results[0][2]
lam = results[0][3]

avg_ml = 0.0
avg_py = 0.0
for i in range(0, 5):
    coefs_ml = HGGM.hggm_matlab(series, int(lag), lam, I_n, I_p, I_g, I_b, eng)
    result_ml = helpers.classify_coefs(coefs_ml, threshold, p)
    fmeasure_ml = helpers.fmeasure(truth, result_ml)
    avg_ml = avg_ml + fmeasure_ml

    coefs_py = HGGM.hggm(series, lam, I_n, I_in, I_p, I_g, I_b, eng, int(lag))[0]
    result_py = helpers.classify_coefs(coefs_py, threshold, p)
    fmeasure_py = helpers.fmeasure(truth, result_py)
    avg_py = avg_py + fmeasure_py

print("Avg matlab F-measure:")
print(avg_ml/10)
print("Avg python F-measure:")
print(avg_py/10)

print("end")