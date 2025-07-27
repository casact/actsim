from actrisk.utils import utils
from actrisk.core.actfitter import DistributionFitter
from actstats import actuarial as act
import numpy as np

# Example usage

# Load data (for example, normally distributed data)
# sev_data = stats.lognorm(0.2, 0, np.exp(0.5)).rvs(size=1000)
sev_data = act.lognormal(0.5,0.2).rvs(size=10000)
freq_data = np.random.poisson(10, 1000)

# Initialize fitter with config file
config = utils.Config('src/actrisk/config.yaml')

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
freq_fitter.summary()