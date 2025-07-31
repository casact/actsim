# ---------------------------------------------
# 1. Import Required Modules
# ---------------------------------------------
from actrisk import StochasticSimulator
from actstats import actuarial as act

# ---------------------------------------------
# 2. Define Frequency and Severity Distributions
# ---------------------------------------------
# Frequency distribution: Poisson with λ=10
freq_dist = 'poisson'
freq_params = (10,)

# Severity distribution: Lognormal with meanlog=10, sigma=0.5
sev_dist = 'lognormal'
sev_params = (10, 0.5)

# Preview quantile (e.g., 80th percentile of Poisson)
quantile_80 = act.poisson.ppf(0.8, 10)
print("80th percentile of Poisson(10):", quantile_80)

# ---------------------------------------------
# 3. Initialize Simulator with Different Levels of Complexity
# ---------------------------------------------

# With copula and correlation settings
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234, 0.6, 'frank', 0.6)

# Without specifying copula_type and theta (defaults apply)
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234, 0.6)

# Without using copula at all
simulator = StochasticSimulator(freq_dist, freq_params, sev_dist, sev_params, 10000, True, 1234)

# ---------------------------------------------
# 4. Generate Simulated Aggregate Losses
# ---------------------------------------------
simulations = simulator.gen_agg_simulations()

# Access full simulation DataFrame
print("All simulations preview:")
print(simulator.all_simulations.head())

# ---------------------------------------------
# 5. Analyze Simulation Results
# ---------------------------------------------

# Calculate aggregate percentile (e.g., 99.2%)
percentile_99_2 = simulator.calc_agg_percentile(99.2)
print("99.2% Aggregate Loss Percentile:", percentile_99_2)

# Plot loss distribution histogram
simulator.plot_distribution()

# Show simulation mean
print("Mean simulated loss:", simulator.results.mean())

# If copula is used, plot frequency-severity correlation structure
simulator.plot_correlated_variables()

# Summary statistics and shape diagnostics
simulator.analyze_results()

# ---------------------------------------------
# 6. Apply Deductibles and Limits
# ---------------------------------------------
# Apply per occurrence deductible of 1,000
# Occurrence limit of 10,000
# Annual aggregate deductible of 100,000
# Annual aggregate limit of 300,000
gross_loss = simulator.apply_deductible_and_limit(1000, 10000, 100000, 300000)


# Assign processed loss to expected structure for reporting
gross_loss['amount'] = gross_loss['gross_loss']

# Re-analyze results based on capped/layered gross loss
simulator.analyze_results(all_simulations=gross_loss)

# ---------------------------------------------
# 7. Export Simulated Data to CSV
# ---------------------------------------------
simulator.all_simulations
