##########################################
###### Synthetic Claim Simulation ########
##########################################
import pandas as pd
import numpy as np
import scipy.stats as stats
from actsim import ClaimSimulator

# Simulate policy characteristics
policies = pd.DataFrame({
    'policy_id': range(1, 11),
    'freq_dist': 'poisson',
    # Frequency parameter (Poisson lambda) per policy as 1-tuples: (20,), (21,), ... (10 values)
    'freq_params': [(20,)] * 10,
    'sev_dist': 'lognormal',
    # Severity parameters per policy (lognormal): first 5 policies use [0.8, 0.3], next 5 use [1.2, 0.7]
    # Use a list-of-tuples so each policy row receives a parameter pair (mu, sigma)
    'sev_params': [(8, 0.3)] * 5 + [(12, 0.7)] * 5,
    'start_date': pd.Timestamp('2023-01-01'),
    'end_date': pd.Timestamp('2023-12-31'),
})

# Instantiate the ClaimSimulator with input policies and np random seed 42
claim_sim = ClaimSimulator(policies, 42)

# Access the processed policy DataFrame
claim_sim.policies

# Run the claim simulation (frequency × severity) for all policy groups
claim_sim.simulate_claims()

# Access the resulting simulated claim records
claim_sim.claim_data

# Set parameters for the non-homogeneous Poisson process (NHPP) for date simulation
lambda0 = 10     # Baseline intensity
alpha = 0.5      # Seasonality amplitude
phase = 0        # Phase shift of the seasonality
T = 1            # Duration of the exposure in years

# Simulate claim occurrence dates using a seasonal NHPP
claim_sim.simulate_dates_nhpp(lambda0, alpha, phase, T)

# Shift claim dates so that the simulation aligns with calendar year starting from 2023
start_year = 2023
claim_sim.apply_shifted_dates(start_year)

# Define base loss development factors (LDFs) by development month
base_LDFs = {
    0: 2,     # Initial LDF at 0 months
    3: 1.5,   # LDF at 3 months
    6: 1.2,
    9: 1.1,
    12: 1.05,
    15: 1.02,
    18: 1.00  # Ultimate LDF at 18 months
}

volatility = 0      # Standard deviation for stochastic fluctuation in LDFs
tail_factor = 1.0     # No additional tail development (fully developed at 18 months)

# Simulate the claim development triangles based on LDFs and apply stochastic volatility
claim_sim.simulate_claim_development(base_LDFs, volatility, tail_factor)

# Access the simulated claim development triangle or long-format development data
claim_sim.claim_development

# Access updated policies
claim_sim.policies

# Access updated claims
claim_sim.claim_data

# Save the simulated claim development data to a file (replace with actual path)
claim_sim.save_claim_development('examples/validation/validation_claim_development.csv')

# ---------------------------------------------
# Validate Distributions
# ---------------------------------------------
claim_development_data = pd.read_csv('examples/validation/validation_claim_development.csv')

# ---------------------------------------------
# Fit Severity Distributions
# ---------------------------------------------

# Validate prescribed severity distributions per policy at development month 18
# For each policy in the `policies` DataFrame, compare observed incurred_loss
# at development_month == 18 to the prescribed `sev_dist` and `sev_params` using KS tests.

alpha = 0.05  # significance level for tests
prescribed_results = []

# Mapping from friendly name to scipy.stats distribution
for _, policy_row in policies.iterrows():
    pid = int(policy_row['policy_id'])
    # We only check lognormal
    sev_params = policy_row['sev_params']

    obs = claim_development_data.loc[
        (claim_development_data['policy_id'] == pid) &
        (claim_development_data['development_month'] == 18),
        'incurred_loss'
    ].dropna()

    if len(obs) < 3:
        prescribed_results.append({
            'policy_id': pid,
            'n_obs': int(len(obs)),
            'prescribed_dist': 'lognormal',
            'prescribed_params': sev_params,
            'test': 'KS',
            'statistic': None,
            'p_value': None,
            'reject_at_0.05': None,
            'note': 'insufficient data'
        })
        continue

    try:
        # Expecting sev_params as (mu, sigma) in log-space
        mu, sigma = sev_params
        # scipy.stats.lognorm: shape = sigma, loc = 0, scale = exp(mu)
        cdf = lambda x: stats.lognorm.cdf(x, s=sigma, loc=0, scale=np.exp(mu))

        data_for_test = obs.values
        data_for_test = data_for_test[np.isfinite(data_for_test)]
        # Lognormal requires positive data
        data_for_test = data_for_test[data_for_test > 0]

        if len(data_for_test) < 3:
            prescribed_results.append({
                'policy_id': pid,
                'n_obs': int(len(obs)),
                'prescribed_dist': 'lognormal',
                'prescribed_params': sev_params,
                'test': 'KS',
                'statistic': None,
                'p_value': None,
                'reject_at_0.05': None,
                'note': 'insufficient positive data for test'
            })
            continue

        ks_stat, p_value = stats.kstest(data_for_test, cdf)

        prescribed_results.append({
            'policy_id': pid,
            'n_obs': int(len(obs)),
            'prescribed_dist': 'lognormal',
            'prescribed_params': sev_params,
            'test': 'KS',
            'statistic': float(ks_stat),
            'p_value': float(p_value),
            'reject_at_0.05': bool(p_value < alpha),
            'note': None
        })
    except Exception as e:
        prescribed_results.append({
            'policy_id': pid,
            'n_obs': int(len(obs)),
            'prescribed_dist': 'lognormal',
            'prescribed_params': sev_params,
            'test': 'KS',
            'statistic': None,
            'p_value': None,
            'reject_at_0.05': None,
            'note': f'test_error: {e}'
        })



# Save and display results
prescribed_df = pd.DataFrame(prescribed_results)
print('\nPrescribed-distribution validation results (development month = 18):')
print(prescribed_df)
prescribed_df.to_csv('examples/validation/validation_prescribed_distribution_check.csv', index=False)