##########################################
###### Synthetic Claim Simulation ########
##########################################
import pandas as pd
import numpy as np
from actrisk import ClaimSimulator 

# Simulate policy characteristics
policies = pd.DataFrame({
    'policy_id': range(1, 101),
    'freq_dist': 'poisson',
    'freq_params': list(zip(np.random.uniform(0.6, 0.8, 100).round(2),)), 
    'sev_dist': 'lognormal',
    'sev_params': list(zip(np.random.uniform(0.8, 1.2, 100).round(2), np.random.uniform(0.3, 0.7, 100).round(2))),
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

volatility = 0.1      # Standard deviation for stochastic fluctuation in LDFs
tail_factor = 1.0     # No additional tail development (fully developed at 18 months)

# Simulate the claim development triangles based on LDFs and apply stochastic volatility
claim_sim.simulate_claim_development(base_LDFs, volatility, tail_factor)

# Access the simulated claim development triangle or long-format development data
claim_sim.claim_development

# Access updated policies (could include mappings to simulated claims)
claim_sim.policies

# Save the simulated claim development data to a file (replace with actual path)
claim_sim.save_claim_development('sample_file_path')