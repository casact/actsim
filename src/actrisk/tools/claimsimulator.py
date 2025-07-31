import pandas as pd
import numpy as np
from typing import Optional
from actrisk.core.actsimulator import StochasticSimulator
from actstats import fraction_to_date_full
from actstats import actuarial as act

class ClaimSimulator:
    def __init__(
        self,
        policies_df: pd.DataFrame,
        random_seed: Optional[int] = 42,
        correlation: Optional[float] = None,
        copula_type: Optional[str] = None,
        copula_param: float = 0
    ):
        self.policies = policies_df.copy()   
        self.keep_all = True     
        self.seed = random_seed
        self.correlation = correlation
        self.copula_type = copula_type
        self.copula_param = copula_param
        self.claim_data = None
        np.random.seed(random_seed)
        self._validate_inputs()

    def _validate_inputs(self):
        required_columns = {
            'policy_id', 'freq_dist', 'freq_params',
            'sev_dist', 'sev_params', 'start_date', 'end_date'
        }
        missing = required_columns - set(self.policies.columns)
        if missing:
            raise ValueError(f"Missing required policy columns: {missing}")

        if not self.policies['freq_params'].apply(lambda x: isinstance(x, tuple)).all():
            raise ValueError("All freq_params must be tuples (e.g., (λ,))")
        if not self.policies['sev_params'].apply(lambda x: isinstance(x, tuple)).all():
            raise ValueError("All sev_params must be tuples (e.g., (mu, sigma))")

    def group_policies(self):
        return self.policies.groupby(['freq_dist', 'sev_dist', 'freq_params', 'sev_params'])

    def simulate_claims(self):
        grouped_policies = self.group_policies()
        simulated_claims = []

        for group_params, group_df in grouped_policies:
            freq_dist = group_params[0]
            sev_dist = group_params[1]
            freq_params = group_params[2]
            sev_params = group_params[3]

            n_policies = len(group_df)

            simulator = StochasticSimulator(
                freq_dist,
                freq_params,
                sev_dist,
                sev_params,
                n_policies,
                self.keep_all,
                self.seed,
                self.correlation,
                self.copula_type,
                self.copula_param
            )

            simulator.gen_agg_simulations()

            if simulator.all_simulations.empty:
                group_claims = pd.DataFrame(columns=['year', 'event_id', 'yearly_event_id', 'amount'])
            else:
                group_claims = simulator.all_simulations.copy()
                group_claims['policy_id'] = group_df['policy_id'].values[group_claims['year']-1]

            simulated_claims.append(group_claims)

        claims_df = pd.concat(simulated_claims, ignore_index=True)
        self.claim_data = claims_df

    def simulate_dates_nhpp(self, lambda0=10, alpha=0.5, phase=0, T=1):
        def simulate_group_dates(group):
            n = len(group)
            nhpp = act.nonhomogeneous_poisson(lambda0, alpha, phase, T)
            fractions = nhpp.rvs(n_events=n)
            year_val = group['year'].iloc[0]
            group = group.copy()
            group['date'] = [fraction_to_date_full(t, year=year_val) for t in fractions]
            return group

        self.claim_data = self.claim_data.groupby('year', group_keys=False).apply(simulate_group_dates)

    @staticmethod
    def shift_date(date_obj, year_shift):
        try:
            return date_obj.replace(year=date_obj.year + year_shift)
        except ValueError:
            return date_obj.replace(month=2, day=28, year=date_obj.year + year_shift)

    def apply_shifted_dates(self, start_year=2023):
        self.claim_data['shifted_date'] = self.claim_data['date'].apply(
            lambda d: self.shift_date(d, start_year)
        )
        self.claim_data.rename(columns={'shifted_date': 'incurred_date', 'amount': 'ultimate_loss'}, inplace=True)
        self.claim_data['incurred_date'] = self.claim_data['incurred_date'].astype('datetime64[s]')

    def simulate_claim_development(self, base_LDFs, volatility=0.1, cumulative_factor=1.0):
        development_data = []

        for _, claim in self.claim_data.iterrows():
            ultimate_loss = claim['ultimate_loss']
            incurred_date = pd.to_datetime(claim['incurred_date'])
            accident_year = incurred_date.year

            unique_LDFs = {dev: ldf * np.random.normal(1, volatility) for dev, ldf in base_LDFs.items()}
            cdf = {}
            cumulative = cumulative_factor
            for dev in sorted(unique_LDFs.keys(), reverse=True):
                cumulative *= unique_LDFs[dev]
                cdf[dev] = cumulative

            for dev_months, cdf_factor in cdf.items():
                reported_loss = ultimate_loss / cdf_factor
                dev_date = incurred_date + pd.DateOffset(months=dev_months)
                development_data.append({
                    'accident_year': accident_year,
                    'incurred_date': claim['incurred_date'],
                    'claim_id': claim['event_id'],
                    'policy_id': claim['policy_id'],
                    'development_month': dev_months,
                    'incurred_loss': reported_loss,
                    'development_date': dev_date
                })

        self.claim_development = pd.DataFrame(development_data)
        self.claim_development = self.claim_development[self.claim_development['accident_year'] < 2200]

    def save_claim_development(self, filepath='examples/reserving_analysis/claim_development_random.csv'):
        self.claim_development.to_csv(filepath, index=False)

