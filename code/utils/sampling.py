import scipy.stats as stats
import numpy as np
import pandas as pd
import code.utils.piecewise_linear_model as pw_linear_model
import matplotlib.pyplot as plt


def calculate_mean_beta(alpha, beta):
    mean = alpha / (alpha + beta)
    return mean

def calculate_sd_beta(alpha, beta):
    sd = np.sqrt(alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    return sd

def calculate_alpha_beta(mean, sd):
    alpha = mean * ((mean * (1 - mean)) / (sd ** 2) - 1)
    return alpha

def calculate_beta_beta(mean, sd):
    beta = (1 - mean) * ((mean * (1- mean)) / (sd **2) - 1)
    return beta

def calc_params_beta(data):
    data_truncated = data[(data > 0)&(data < 1)].reset_index(drop = True)
    a, b, loc, scale = stats.beta.fit(data_truncated, floc = 0, fscale = 1)
    mean = calculate_mean_beta(a, b)
    sd = calculate_sd_beta(a, b)
    return pd.Series([a, b , mean, sd], index = ['alpha', 'beta', 'mean', 'std'])

def tranform_2nd_polynomial(series, a, b, c):
    transformed_series = a * series**2 + b * series + c
    return transformed_series

def calc_sampling_dis_params(data:pd.DataFrame, input):
    ####### proportion of 0 #########
    startpoint = 0.8
    breakpoint = [0, 0.2, 0.6, 1]
    slopes = [-0.3, -0.5, 0]
    linear_model = pw_linear_model.PiecewiseLinearModel(startpoint, breakpoint, slopes)
    vector_0_prop = linear_model.apply_to_column(data[input])

    ####### proportion of 1 #########
    startpoint = 0
    breakpoint = [0, 0.2, 0.4, 1]
    slopes = [0, 0.25, 1.25]
    linear_model = pw_linear_model.PiecewiseLinearModel(startpoint, breakpoint, slopes)
    vector_1_prop = linear_model.apply_to_column(data[input])

    ###### beta distribution ########
    [a, b, c] = [-0.76, 0.65, 0.036]

    # recalibration of the mean would result in mean greater than 100%
    # init_mean = data[input]
    # mean = (init_mean - vector_1_prop * 1)/(1 - vector_0_prop - vector_1_prop)
    mean = data[input]
    sd = tranform_2nd_polynomial(mean, a, b, c)
    alpha = calculate_alpha_beta(mean, sd)
    beta = calculate_beta_beta(mean, sd)

    df_params = pd.DataFrame({
        'vector_0_prop': vector_0_prop,
        'vector_1_prop': vector_1_prop,
        'alpha': alpha,
        'beta': beta
    })

    return df_params

def beta_sampling(alpha, beta, zero_prop, one_prop, n: int):
    num_0 = int(zero_prop * n)
    num_1 = int(one_prop * n)
    num_beta = n - num_0 - num_1
    beta_sample = stats.beta.rvs(alpha, beta, size = num_beta)
    sample = np.concatenate((np.zeros(num_0), np.ones(num_1), beta_sample))
    np.random.shuffle(sample)
    return pd.Series(sample)

####### example ########
sev_data = pd.Series([0.2, 0.3, 0.5, 0.4, 0.4, 0.6])
sev_data = pd.Series(np.random.beta(2, 4, 1000))
a, b , mean, sd = calc_params_beta(sev_data)

data = pd.DataFrame({'mean':sev_data})
sample_params = calc_sampling_dis_params(data, 'mean')
simulated_sample = sample_params.apply(lambda row:beta_sampling(row['alpha'], row['beta'], row['vector_0_prop'], row['vector_1_prop'], n = 100), axis = 1)
row = simulated_sample.iloc[0]
df = 1 - sample_params['vector_0_prop'] - sample_params['vector_1_prop']
df = df < 0
df.value_counts()

plt.hist(row, bins=50, density=True, alpha=0.5, color='g')
plt.title('Distribution of Simulated Aggregate Losses')
plt.xlabel('Aggregate Loss')
plt.ylabel('Density')
plt.show()
