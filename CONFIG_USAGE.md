# Using the Config File After Package Installation

After installing the `actrisk` package, you can use the config file in several ways:

## Method 1: Use the Default Config (Recommended)

```python
from actrisk import DistributionFitter, load_config

# Load the default config file
config = load_config()

# Use the config values
severity_distributions = config.distributions['severity']
frequency_distributions = config.distributions['frequency']
metrics = config.metrics

# Create a fitter with config values
fitter = DistributionFitter(
    data=your_data,
    distributions=severity_distributions,
    metrics=metrics
)
```

## Method 2: Use Config Class Directly

You can also use the Config class directly:

```python
from actrisk import DistributionFitter, Config

# Use default config
config = Config()

# Or specify a custom config file
config = Config('path/to/your/custom_config.yaml')
```

## Method 3: Import from Utils

For more advanced usage:

```python
from actrisk.utils import Config
from actrisk.core.actfitter import DistributionFitter

config = Config()  # Uses default config
```

## Default Config Contents

The default config file includes:

```yaml
distributions:
  severity:
    - normal
    - logistic
    - exponential
    - gamma
    - beta
    - lognormal
    - weibull
    - pareto
    - uniform
  frequency:
    - poisson
    - negative binomial

metrics:
  - aic
  - bic
  - log_likelihood
  - chisquare
```

## Custom Config Files

You can create your own config file and use it:

```python
# custom_config.yaml
distributions:
  severity:
    - normal
    - lognormal
    - gamma
  frequency:
    - poisson

metrics:
  - aic
  - bic

# In your code
config = load_config('custom_config.yaml')
```

## Config Object Methods

The Config object provides several useful methods:

```python
config = load_config()

# Check if a key exists
if config.has_key('distributions'):
    print("Distributions section found")

# Update config values
config.update({'new_metric': 'ks_test'})

# Reload from file
config.reload()

# Access values as attributes
print(config.distributions['severity'])
```

## Complete Example

```python
from actrisk import DistributionFitter, load_config
import numpy as np

# Load config
config = load_config()

# Generate sample data
sev_data = np.random.lognormal(0.5, 0.2, size=1000)

# Fit severity using config distributions and metrics
sev_fitter = DistributionFitter(
    sev_data, 
    distributions=config.distributions['severity'], 
    metrics=config.metrics
)

# Fit the distributions
sev_fitter.fit()

# Get best fit based on AIC
best_fit = sev_fitter.get_best_fit('aic')
print(f"Best distribution: {best_fit['name']}")
```
