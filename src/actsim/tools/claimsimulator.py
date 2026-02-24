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
        self.rng = np.random.default_rng(random_seed) # Limit use of random seed to current code
        self.correlation = correlation
        self.copula_type = copula_type
        self.copula_param = copula_param
        self.claim_data = None
        np.random.seed(random_seed)
        self._validate_inputs()

    # ------------------------------------------------------------------ #
    #  Validation & preparation                                            #
    # ------------------------------------------------------------------ #

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

    def _prepare_policy_dates(self) -> None:
        """Parse and validate policy date columns; add convenience columns."""
        self.policies["start_date"] = pd.to_datetime(self.policies["start_date"])
        self.policies["end_date"] = pd.to_datetime(self.policies["end_date"])

        invalid = self.policies[self.policies["end_date"] <= self.policies["start_date"]]
        if not invalid.empty:
            raise ValueError(
                f"end_date must be after start_date for policies: "
                f"{invalid['policy_id'].tolist()}"
            )

        # Fractional exposure within each calendar year [0, 1]
        self.policies["_year_start_frac"] = self.policies["start_date"].apply(
            lambda d: self._date_to_year_fraction(d)
        )
        self.policies["_year_end_frac"] = self.policies["end_date"].apply(
            lambda d: self._date_to_year_fraction(d)
        )
        # Exposure in years (used for frequency scaling)
        self.policies["_exposure_years"] = (
            (self.policies["end_date"] - self.policies["start_date"]).dt.days / 365.25
        )
    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _date_to_year_fraction(d: pd.Timestamp) -> float:
        """Return the fraction of the year elapsed at date d (0 = Jan 1, ~1 = Dec 31)."""
        start_of_year = pd.Timestamp(year=d.year, month=1, day=1)
        start_of_next = pd.Timestamp(year=d.year + 1, month=1, day=1)
        return (d - start_of_year) / (start_of_next - start_of_year)

    @staticmethod
    def _shift_date(date_obj: pd.Timestamp, year_shift: int) -> pd.Timestamp:
        try:
            return date_obj.replace(year=date_obj.year + year_shift)
        except ValueError:  # Feb 29 in non-leap year
            return date_obj.replace(month=2, day=28, year=date_obj.year + year_shift)

    # ------------------------------------------------------------------ #
    #  Simulation: claim counts and severities                            #
    # ------------------------------------------------------------------ #

    def group_policies(self):
        return self.policies.groupby(['freq_dist', 'sev_dist', 'freq_params', 'sev_params'], sort=False)

    def simulate_claims(self):
        grouped_policies = self.group_policies()
        simulated_claims = []

        for group_params, group_df in grouped_policies:
            freq_dist, sev_dist, freq_params, sev_params = group_params
            group_df = group_df.reset_index(drop=True)
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
                continue

            group_claims = simulator.all_simulations.copy()

            # 'year' in the simulator output is a 1-based policy index within
            # this group — map it to the actual policy metadata
            policy_index = group_claims["year"] - 1  # 0-based
            group_claims["policy_id"] = group_df.loc[policy_index, "policy_id"].values
            group_claims["start_date"] = group_df.loc[
                policy_index, "start_date"
            ].values
            group_claims["end_date"] = group_df.loc[
                policy_index, "end_date"
            ].values

            simulated_claims.append(group_claims)

        if not simulated_claims:
            self.claim_data = pd.DataFrame(
                columns=[
                    "year", "event_id", "yearly_event_id", "amount",
                    "policy_id", "start_date", "end_date",
                ]
            )
            return

        self.claim_data = pd.concat(simulated_claims, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Date simulation                                                     #
    # ------------------------------------------------------------------ #

    def simulate_dates_nhpp(
        self, lambda0: float = 10, alpha: float = 0.5, phase: float = 0, T: float = 1
    ) -> None:
        """
        Assign an incurred date to each claim using a Non-Homogeneous Poisson
        Process, constrained to each policy's active window within its year.
        """
        if self.claim_data is None or self.claim_data.empty:
            raise RuntimeError("Call simulate_claims() before simulate_dates_nhpp().")

        nhpp = act.nonhomogeneous_poisson(lambda0, alpha, phase, T)

        def _assign_dates(row: pd.Series) -> pd.Timestamp:
            policy_start = pd.Timestamp(row["start_date"])
            policy_end = pd.Timestamp(row["end_date"])

            # Draw a fraction in [0, 1] from the NHPP and scale to the policy window
            frac = nhpp.rvs(n_events=1)[0]
            policy_duration = (policy_end - policy_start).days
            claim_date = policy_start + pd.Timedelta(days=frac * policy_duration)

            # Clamp to policy window (defensive)
            return max(policy_start, min(claim_date, policy_end))

        self.claim_data["incurred_date"] = self.claim_data.apply(
            _assign_dates, axis=1
        )
        self.claim_data["incurred_date"] = self.claim_data["incurred_date"].astype(
            "datetime64[s]"
        )
        # Clean up intermediary columns kept only for date simulation
        self.claim_data.drop(
            columns=["start_date", "end_date"], inplace=True, errors="ignore"
        )
        self.claim_data.rename(columns={"amount": "ultimate_loss"}, inplace=True)

    # ------------------------------------------------------------------ #
    #  Claim development (triangle)                                        #
    # ------------------------------------------------------------------ #

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
