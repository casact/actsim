import yaml
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import wraps
import actrisk.utils.utils as utils
import pandas as pd
import actrisk.core.actsimulator as stk
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
                # ks_statistic = stats.kstest(self.data, distribution.cdf, args=params).statistic

                result = {
                    'name': name,
                    'distribution': distribution,
                    'params': params,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'chisquare': chi_square,
                    #'ks': ks_statistic
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

if __name__ == "__main__":
    # Example usage

    # Load data (for example, normally distributed data)
    # sev_data = stats.lognorm(0.2, 0, np.exp(0.5)).rvs(size=1000)
    sev_data = act.lognormal(0.5,0.2).rvs(size=10000)
    freq_data = np.random.poisson(10, 1000)

    # Initialize fitter with config file
    config = utils.Config('code/config.yaml')

    #############################
    ###### Fit Severity #########
    #############################
    # User specifies distributions and metrics 
    distribution_names = config.distributions['severity']
    metrics = config.metrics

    sev_fitter = DistributionFitter(sev_data, distributions=distribution_names, metrics=metrics)
    sev_fitter.fit()
    sev_fitter.best_fits
    sev_fitter.selected_fit
    sev_fitter.get_selected_dist()
    # Selecting a distribution manually
    sev_fitter.select_distribution('uniform')
    selected_fit = sev_fitter.selected_fit

    print("Selected fitting distribution:", selected_fit['name'])
    print("Parameters:", selected_fit['params'])
    print("AIC:", selected_fit['aic'])
    print("BIC:", selected_fit['bic'])

    # Calculating statistics
    sev_fitter.calculate_statistics().to_csv('outputs/statistics.csv')

    # Plotting predictions
    sev_fitter.plot_predictions()

    # Produce summary
    sev_fitter.summary().to_csv('outputs/summary.csv')

    # Generating samples
    samples = sev_fitter.sample(size=10)
    print("Generated samples:", samples)

    samples = sev_fitter.sample_mixed(0.1, 0.1, size=10)

    #############################
    ###### Fit frequency ########
    #############################
    distribution_names = config.distributions['frequency']
    metrics = config.metrics

    freq_fitter = DistributionFitter(freq_data, distributions=distribution_names, metrics=metrics)
    freq_fitter.distributions
    freq_fitter.fit()
    freq_fitter.best_fits
    freq_fitter.selected_fit


    #####################################
    ###### Stochastic Simulation ########
    #####################################
    freq_dist = freq_fitter.selected_fit['name']
    freq_params = freq_fitter.get_selected_params()
    sev_dist = sev_fitter.selected_fit['name']
    sev_params = sev_fitter.get_selected_params()


    simulator = stk.StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 100, True, 1234, 0.6, 'frank', 0.6)
    simulator = stk.StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 100, True, 1234, 0.6)
    simulator = stk.StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 100, True, 1234)

    simulations = simulator.gen_agg_simulations()
    simulator.all_simulations
    simulator.calc_agg_percentile(99.2)
    simulator.plot_distribution()
    simulator.results.mean()
    simulator.plot_correlated_variables()
    simulator.all_simulations
    print(pd.DataFrame(simulator.analyze_results()))

    ##### Generate correlated mutivariate distribution
    corr_matrix_file = 'code/utils/corr_matrix.csv'
    dist_list_file = 'code/utils/dist_list.json'
    simulator = stk.StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1034, 0.6)
    simulator.gen_multivariate_corr_simulations(corr_matrix_file, dist_list_file, True)
    simulator._all_simulations_data
    data = pd.DataFrame(simulator._all_simulations_data)
    data_t = data.transpose()
    # Compute correlation matrix
    correlation_matrix = data_t.corr()
    print(correlation_matrix)