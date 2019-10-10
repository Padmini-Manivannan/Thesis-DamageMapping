"""
To study the general distribution of the amplitudes of Persistent Scatterers.
"""
from Damage_mapping.PS_mapping import ps_amp
# from Damage_mapping.AmplitudeAnalysis import ps_amp

import numpy as np
# import warnings
import scipy.stats as st
# import statsmodels as sm
import matplotlib.pyplot as plt


# matplotlib.rcParams['figure.figsize'] = (7.0, 5.0)
# matplotlib.style.use('ggplot')

# Import the PS Amplitudes file
# ps_amp = np.loadtxt('PSC_Amplitudes_22_26.csv', delimiter=',')
# ps_amp = np.transpose(ps_amp)
# Distributions to check
DISTRIBUTIONS = [
    st.rayleigh, st.rice, st.norm
]

# realdata = ps_amp[:, :, 0]
realdata = ps_amp

# Histogram of data
for i in range(10):
    plt.figure()
    plt.hist(realdata[:-1, i], density=True)
    plt.show()

plt.close('all')

# Two sample KS test for Rice distributions as reference - This works
b = 1
resultsRice = np.zeros((realdata.shape[1], 2))
for i in range(realdata.shape[1]):
    data = realdata[:-1, i]
    loc = data.mean()
    scale = data.std()
    ref = st.rice.rvs(b, loc=loc, scale=scale, size=24)
    d, p = st.ks_2samp(data, ref)
    resultsRice[i, 0] = d
    resultsRice[i, 1] = p
print(p.max())

ks_plot(data)

""" 
d = KS distance, supremum of the set of distances betweent the cdfs
p = some value describing similarity of both cdfs 

The p value is extremely low for almost all of the PSs. 
The lower your p value the greater the statistical evidence 
you have to reject the null hypothesis and conclude the distributions are different. 

Highest p value : 0.00020211742020648208
Hence, we cannot conclude that the sample data is Rice distributed.
"""

# Check if results are Rayleigh distributed
resultsRay = np.zeros((realdata.shape[1], 2))
for i in range(realdata.shape[1]):
    data = realdata[:-1, i]
    loc = data.mean()
    scale = data.std()
    ref = st.rayleigh.rvs(loc=loc, scale=scale, size=30)
    d, p = st.ks_2samp(data, ref)
    resultsRay[i, 0] = d
    resultsRay[i, 1] = p
print(p.max())

"""
Not Rayleigh distributed either. 
Highest p value: 0.004607065432753834
"""

# Check if the data is Gaussian distributed - using Anderson-Darling Test
# resultsNorm = np.zeros((ps_amp.shape[1], 6))
H0 = 0
H1 = 0
for i in range(realdata.shape[1]):
    data = realdata[:-1, i]
    resultsNorm = st.anderson(data, 'norm')
    # resultsNorm[i, 0] = st.anderson(data, 'norm').statistic
    # resultsNorm[i, 1:] = st.anderson(data, 'norm').critical_values
    sl, cv = resultsNorm.significance_level[-1], resultsNorm.critical_values[-1]
    if resultsNorm.statistic < resultsNorm.critical_values[-1]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        H0 += 1
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        H1 += 1
    # With significance_level = array([15., 10., 5., 2.5, 1.]))
    #  25%, 10%, 5%, 2.5%, 1%.

"""
The idea is that if the returned statistic is larger than these critical values,
then for the corresponding significance level, the null hypothesis that the data come 
from the chosen distribution can be rejected.

We can interpret the results by failing to reject the null hypothesis that the data is normal 
if the calculated test statistic is less than the critical value at a chosen significance level.

The level of significance is defined as the probability of rejecting a null hypothesis by the test 
when it is really true, which is denoted as α. That is, P (Type I error) = α.
Here, considering a significance level of 1% or a confidence level of 99%

Most of the PSs (289, 260) are Gaussian-like and some (54, 46) are not. About 84% is gaussian.
According to Ferretti, for high SNR, values approach a Gaussian distribution.

Most of the spatially-averaged amplitudes in the area of interest are Gaussian (18739) and the rest (1261) are not. 
About 93.6% are Gaussian. 


"""

