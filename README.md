# ActRisk

[![PyPI version](https://badge.fury.io/py/actrisk.svg)](https://badge.fury.io/py/actrisk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for actuarial risk modeling and simulation.

## Features

- **Risk Modeling**: Advanced tools for actuarial risk analysis
- **Monte Carlo Simulations**: High-performance simulation capabilities
- **Parallel Processing**: Optimized for large-scale computations
- **Configuration Management**: Flexible YAML-based configuration system
- **Statistical Analysis**: Comprehensive statistical tools for risk assessment

## Installation

### From PyPI (recommended)

```bash
pip install actrisk
```

### From Source

```bash
git clone https://github.com/yourusername/actrisk.git
cd actrisk
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/actrisk.git
cd actrisk
pip install -e .[dev]
```

## Quick Start

```python
from actrisk.core import actfitter
from actstats import actuarial as act

# Load configuration
config = Config('config.yaml')

sev_data = act.lognormal(0.5,0.2).rvs(size=10000)

#############################
###### Fit Severity #########
#############################
# User specifies distributions and metrics 
distribution_names = config.distributions['severity']
metrics = config.metrics

sev_fitter = actfitter(sev_data, distributions=distribution_names, metrics=metrics)
sev_fitter.fit()
sev_fitter.best_fits
sev_fitter.selected_fit
```

## Documentation

- [User Guide](docs/user_guide.md) - Getting started and basic usage
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Examples](docs/examples.md) - Code examples and tutorials
- [Development](docs/development.md) - Contributing and development guidelines

## Features in Detail

### Configuration Management
```python
# Initialize fitter with config file
config = utils.Config('code/config.yaml')

```

### Fit Severity
```python
sev_data = act.lognormal(0.5,0.2).rvs(size=10000)

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
```

### Fit Frequency

```python
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

```

### Stochastic Simulation

```python
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
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/actrisk.git
cd actrisk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ActRisk in your research, please cite:

```bibtex
@software{actrisk2025,
  title={ActRisk: A Python package for actuarial risk modeling and simulation},
  author={Juntao Zhang},
  year={2025},
  url={https://github.com/jzhng105/actrisk}
}
```

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/jzhng105/actrisk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jzhng105/actrisk/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.