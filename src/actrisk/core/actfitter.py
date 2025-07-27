import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import wraps
import pandas as pd
from actstats import actuarial as act

# Decorator to check if a distribution has been selected
def check_selected_dist(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.selected_fit is None:
            raise ValueError("No distribution has been selected yet. Use 'select_distribution' method first.")
        return func(self, *args, **kwargs)
    return wrapper

class DistributionFitter:
    def __init__(self, data, distributions=None, metrics=None):
        self.data = data
        self._length = len(data)
        self.available_distributions = {
            'uniform': act.uniform,
            'normal': act.normal,
            'logistic': act.logistic,
            'exponential': act.exponential,
            'gamma': act.gamma,
            'beta': act.beta,
            'pareto': act.pareto,
            'poisson': act.poisson,
            'weibull': act.weibull,
            'lognormal': act.lognormal,
            'negative binomial': act.negative_binomial,
        }

        # Filter available distributions based on user inputs
        if distributions:
            self.distributions = {name: self.available_distributions[name] for name in distributions if name in self.available_distributions}
        else:
            self.distributions = self.available_distributions

        self.metrics = metrics if metrics else ['aic', 'bic']

        self.results = []
        self.best_fits = {} 
        self.statistics = {}
        self.selected_fit = None
    
    def truncate_data(self, i=int):
        self.data = self.data[(self.data != i).all(axis=1)]

    def fit(self):
        if self.data is None:
            raise ValueError("No data has been loaded. Use 'load_data' method first.")
        
        for name, distribution in self.distributions.items():
            try:
                params = distribution.fit(self.data)
                print(f"{name}: {params}")
                if name == 'poisson' or name == 'negative binomial':
                    log_likelihood = np.sum(distribution(*params).logpmf(self.data))
                else:
                    log_likelihood = np.sum(distribution(*params).logpdf(self.data))
                # AIC
                aic = 2 * len(params) - 2 * log_likelihood

                # BIC
                bic = np.log(len(self.data)) * len(params) - 2 * log_likelihood

                # Chi-square test with normalization
                expected_freq, _ = np.histogram(self.data, bins=10, density=False)
                observed_sample = distribution(*params).rvs(size=len(self.data))
                observed_freq, _ = np.histogram(observed_sample, bins=10)

                # Normalize observed frequencies to match the sum of expected frequencies
                observed_freq = observed_freq * (expected_freq.sum() / observed_freq.sum())

                # Perform Chi-square test
                chi_square = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq).statistic

                # K-S test
                ks_statistic = stats.kstest(self.data, distribution.cdf).statistic

                result = {
                    'name': name,
                    'distribution': distribution,
                    'params': params,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'chisquare': chi_square,
                    'ks': ks_statistic
                }

                self.results.append(result)
            except Exception as e:
                print(f"Could not fit {name} distribution: {e}")

        self.select_best_fit()

    def select_best_fit(self):
        if not self.results:
            raise ValueError("No distributions have been fitted yet. Call the 'fit' method first.")

        best_fit = None
        for metric in self.metrics:
            best_fit = min(self.results, key=lambda x: x[metric])
            self.best_fits[metric] = best_fit
        
        self.selected_fit = self.best_fits['aic']  # Default selected fit best fit under AIC 
    
    def get_best_fit(self, metric):
        """Get the best-fitting distribution for a specific metric."""
        return self.best_fits.get(metric, None)

    def select_distribution(self, name):
        # Next () retrieves the first result that meets the condition.
        match = next((result for result in self.results if result['name'] == name), None)
        if match is None:
            raise ValueError(f"No distribution named '{name}' found in the fitted results.")
        self.selected_fit = match

    @check_selected_dist
    def get_selected_dist(self):
        return self.selected_fit['distribution']
    
    @check_selected_dist
    def get_selected_params(self):
        return self.selected_fit['params']

    @check_selected_dist
    def predict(self, x):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        return distribution.pdf(x, *params)

    @check_selected_dist
    def sample(self, size=1):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        return distribution.rvs(*params, size=size)
    
    @check_selected_dist
    def sample_mixed(self, zero_prop=0, one_prop=0, size=1):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        num_0 = int(zero_prop * size)
        num_1 = int(one_prop * size)
        num_sample = size - num_0 - num_1
        mixed_sample = distribution.rvs(*params, size=num_sample)
        sample = np.concatenate((np.zeros(num_0), np.ones(num_1), mixed_sample))
        np.random.shuffle(sample)
        return pd.Series(sample)

    @check_selected_dist
    def calculate_statistics(self):
        # Data statistics
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        data_percentiles = np.percentile(self.data, [5, 25, 50, 75, 95])

        # Predicted statistics
        x_values = np.linspace(min(self.data), max(self.data), len(self.data))
        predicted_pdf = self.predict(x_values)
        predicted_mean = np.sum(x_values * predicted_pdf) / np.sum(predicted_pdf)
        predicted_std = np.sqrt(np.sum((x_values - predicted_mean)**2 * predicted_pdf) / np.sum(predicted_pdf))
        predicted_percentiles = np.percentile(predicted_pdf, [5, 25, 50, 75, 95])

        self.statistics =  {
            'data': {
                'mean': data_mean,
                'std': data_std,
                'percentiles': data_percentiles
            },
            'predicted': {
                'mean': predicted_mean,
                'std': predicted_std,
                'percentiles': predicted_percentiles
            }
        }

        return pd.DataFrame(self.statistics)

    def plot_predictions(self, distribution_names=None):
        """Plot the data and the PDFs of selected distributions."""
        if distribution_names is None:
            distribution_names = [result['name'] for result in self.results]

        # Get colors for the distributions
        colors = mpl.colormaps.get_cmap('tab10')
        
        x_values = np.linspace(min(self.data), max(self.data), 100)
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of the data
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='gray', label='Actual Data')

        # Plot the PDFs of the selected distributions
        for idx, name in enumerate(distribution_names):
            result = next((result for result in self.results if result['name'] == name), None)
            if result:
                pdf_values = result['distribution'].pdf(x_values, *result['params'])
                plt.plot(x_values, pdf_values, lw=2, label=f'{name} PDF', color=colors(idx))

        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('Fitted Distributions vs Actual Data')
        plt.legend()
        plt.show()


    def summary(self):
        if not self.results:
            raise ValueError("No distributions have been fitted yet. Call the 'fit' method first.")
        return pd.DataFrame(self.results)
