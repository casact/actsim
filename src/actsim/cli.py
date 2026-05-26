"""Command-line interface for the actsim package."""
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ALL_FIT_METRICS = ["aic", "bic", "log_likelihood", "chisquare", "ks"]


def _parse_params(raw: str) -> tuple:
    """Parse a comma-separated parameter string into a tuple of numbers."""
    if raw is None or raw == "":
        return ()
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Could not parse parameter value: {p!r}")
        out.append(int(v) if v.is_integer() else v)
    return tuple(out)


def _load_series(path: Path, column: str | None) -> pd.Series:
    """Load a one-dimensional numeric series from CSV or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise SystemExit(f"Unsupported input format: {suffix} (use .csv or .json)")

    if column:
        if column not in df.columns:
            raise SystemExit(f"Column {column!r} not found. Available: {list(df.columns)}")
        return pd.to_numeric(df[column], errors="coerce").dropna()

    if df.shape[1] != 1:
        raise SystemExit(
            f"Input has {df.shape[1]} columns; pass --column to pick one. "
            f"Available: {list(df.columns)}"
        )
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()


def _write_or_print_df(df: pd.DataFrame, output: Path | None, label: str = "rows") -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Wrote {len(df)} {label} to {output}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(df.to_string(index=False))


def _tvar(values: np.ndarray, q: float) -> float:
    """Tail Value-at-Risk: mean of losses strictly above the q-quantile."""
    cutoff = np.quantile(values, q)
    tail = values[values > cutoff]
    return float(tail.mean()) if tail.size else float(cutoff)


def _coerce_param_cell(value):
    """Best-effort coercion of a CSV cell into a tuple of numbers."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    s = str(value).strip()
    if s.startswith(("(", "[")) and s.endswith((")", "]")):
        s = s[1:-1]
    return _parse_params(s)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_version(_args: argparse.Namespace) -> int:
    try:
        print(metadata.version("actsim"))
    except metadata.PackageNotFoundError:
        print("unknown")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    from . import load_config

    cfg = load_config(args.config)
    print(json.dumps(cfg._data, indent=2, default=str))
    return 0


def cmd_config_path(_args: argparse.Namespace) -> int:
    from . import load_config

    cfg = load_config()
    print(cfg._file_path)
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from . import DistributionFitter, load_config

    data = _load_series(Path(args.input), args.column)

    cfg = load_config(args.config) if args.config else load_config()
    severity_default = (cfg.distributions or {}).get("severity") if cfg._data else None

    distributions = args.distributions or severity_default
    # Always compute every available metric — they're cheap and 'ks' is
    # otherwise hidden even though the fitter computes it for every model.
    metrics = args.metrics or ALL_FIT_METRICS

    fitter = DistributionFitter(
        data=data, distributions=distributions, metrics=metrics
    )

    if any(
        v is not None
        for v in (
            args.truncate_remove, args.truncate_lower, args.truncate_upper,
            args.truncate_q_low, args.truncate_q_high,
        )
    ):
        remove_values = (
            _parse_params(args.truncate_remove) if args.truncate_remove else None
        )
        fitter.truncate_data(
            remove_values=list(remove_values) if remove_values else None,
            lower=args.truncate_lower,
            upper=args.truncate_upper,
            q_low=args.truncate_q_low,
            q_high=args.truncate_q_high,
        )
        print(f"Truncated to {len(fitter.data)} rows.")

    fitter.fit()

    cols = ["name", "aic", "bic", "log_likelihood", "chisquare", "ks", "params"]
    summary = fitter.summary()[cols].sort_values(args.metric)
    print()
    print(f"Fit summary (sorted by {args.metric}):")
    _write_or_print_df(summary, Path(args.output) if args.output else None, "fits")

    best = fitter.get_best_fit(args.metric)
    if best is not None:
        print()
        print(f"Best fit by {args.metric}: {best['name']}  params={best['params']}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Fit, then draw random samples from the chosen distribution."""
    from . import DistributionFitter, load_config

    data = _load_series(Path(args.input), args.column)
    cfg = load_config()
    severity_default = (cfg.distributions or {}).get("severity") if cfg._data else None
    distributions = args.distributions or severity_default

    fitter = DistributionFitter(
        data=data, distributions=distributions, metrics=["aic", "bic"]
    )
    fitter.fit()

    if args.distribution:
        fitter.select_distribution(args.distribution)
    chosen = fitter.selected_fit
    print(f"Sampling from {chosen['name']} (params={chosen['params']}).")

    if args.zero_prop or args.one_prop:
        draws = fitter.sample_mixed(
            zero_prop=args.zero_prop, one_prop=args.one_prop, size=args.size
        )
    else:
        draws = fitter.sample(size=args.size)

    out_df = pd.DataFrame({"sample": np.asarray(draws)})
    _write_or_print_df(
        out_df.head(args.size if args.output else 20),
        Path(args.output) if args.output else None,
        "samples",
    )
    print()
    print(
        f"Mean: {out_df['sample'].mean():,.4f}    "
        f"Std: {out_df['sample'].std():,.4f}    "
        f"Min: {out_df['sample'].min():,.4f}    "
        f"Max: {out_df['sample'].max():,.4f}"
    )
    return 0


def _apply_layer(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame | None:
    """Apply per-occurrence + aggregate deductible/limit to event-level data."""
    if not any(
        v is not None
        for v in (args.per_occ_ded, args.per_occ_limit, args.agg_ded, args.agg_limit)
    ):
        return None

    df = events.copy()
    per_occ_ded = args.per_occ_ded or 0.0
    per_occ_limit = args.per_occ_limit if args.per_occ_limit is not None else np.inf
    agg_ded = args.agg_ded or 0.0
    agg_limit = args.agg_limit if args.agg_limit is not None else np.inf

    df["gross_loss"] = (df["amount"].clip(lower=per_occ_ded) - per_occ_ded).clip(
        upper=per_occ_limit
    )
    annual = df.groupby("year", as_index=False)["gross_loss"].sum()
    annual["net_loss"] = (annual["gross_loss"].clip(lower=agg_ded) - agg_ded).clip(
        upper=agg_limit
    )
    return annual


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import StochasticSimulator

    sim = StochasticSimulator(
        freq_dist=args.freq_dist,
        freq_params=_parse_params(args.freq_params),
        sev_dist=args.sev_dist,
        sev_params=_parse_params(args.sev_params),
        num_sim=args.num_sim,
        keep_all=args.keep_all,
        seed=args.seed,
        correlation=args.correlation,
        copula_type=args.copula,
        theta=args.theta,
    )
    sim.gen_agg_simulations()

    quantiles = args.quantiles or [0.5, 0.75, 0.9, 0.95, 0.99]
    results_arr = np.asarray(sim.results)
    summary: dict[str, list[float]] = {
        "VaR": [np.quantile(results_arr, q) for q in quantiles],
        "TVaR": [_tvar(results_arr, q) for q in quantiles],
        "AEP": [np.quantile(results_arr, q) for q in quantiles],
    }
    if args.keep_all:
        events = sim.all_simulations
        if not events.empty:
            max_per_year = np.sort(events.groupby("year")["amount"].max().values)
            summary["OEP"] = [
                max_per_year[min(int(q * len(max_per_year)), len(max_per_year) - 1)]
                for q in quantiles
            ]
    report = pd.DataFrame(summary, index=quantiles).round(2)
    print("Aggregate loss summary:")
    print(report.to_string())

    print()
    print(
        f"Mean: {np.mean(sim.results):,.2f}    "
        f"Std: {np.std(sim.results):,.2f}    "
        f"Max: {np.max(sim.results):,.2f}"
    )

    layered = None
    if args.keep_all:
        layered = _apply_layer(sim.all_simulations, args)
        if layered is not None:
            print()
            print("Layered (per-occ + aggregate ded/limit) summary:")
            print(
                f"  Gross mean:  {layered['gross_loss'].mean():,.2f}    "
                f"Net mean:    {layered['net_loss'].mean():,.2f}"
            )
            layer_q = pd.DataFrame(
                {
                    "Gross VaR": [np.quantile(layered["gross_loss"], q) for q in quantiles],
                    "Net VaR": [np.quantile(layered["net_loss"], q) for q in quantiles],
                    "Net TVaR": [_tvar(layered["net_loss"].values, q) for q in quantiles],
                },
                index=quantiles,
            ).round(2)
            print(layer_q.to_string())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if layered is not None:
            layered.to_csv(out_path, index=False)
            print(f"Wrote layered annual data to {out_path}")
        elif args.keep_all:
            sim.all_simulations.to_csv(out_path, index=False)
            print(f"Wrote per-event data to {out_path}")
        else:
            sim.results.to_frame("aggregate_loss").to_csv(out_path, index=False)
            print(f"Wrote aggregate simulations to {out_path}")
    return 0


def cmd_simulate_corr(args: argparse.Namespace) -> int:
    from . import StochasticSimulator

    sim = StochasticSimulator(
        freq_dist="poisson",
        freq_params=(1,),
        sev_dist="normal",
        sev_params=(0, 1),
        num_sim=args.num_sim,
        keep_all=args.keep_all,
        seed=args.seed,
    )
    aggregate = sim.gen_multivariate_corr_simulations(
        corr_matrix_file=args.corr_matrix,
        dist_list_file=args.dist_list,
        gen_marginal=args.keep_all,
    )

    quantiles = args.quantiles or [0.5, 0.75, 0.9, 0.95, 0.99]
    summary = pd.DataFrame(
        {
            "VaR": [np.quantile(aggregate, q) for q in quantiles],
            "TVaR": [_tvar(np.asarray(aggregate), q) for q in quantiles],
        },
        index=quantiles,
    ).round(2)
    print("Multivariate aggregate summary:")
    print(summary.to_string())
    print()
    print(
        f"Mean: {np.mean(aggregate):,.2f}    "
        f"Std: {np.std(aggregate):,.2f}    "
        f"Max: {np.max(aggregate):,.2f}"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.keep_all:
            dist_names = [d["dist_name"] for d in json.loads(Path(args.dist_list).read_text())]
            df = pd.DataFrame(sim._all_simulations_data.T, columns=dist_names)
            df["aggregate"] = aggregate
            df.to_csv(out_path, index=False)
        else:
            pd.Series(aggregate, name="aggregate").to_frame().to_csv(out_path, index=False)
        print(f"Wrote results to {out_path}")
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from . import ClaimSimulator

    policies = pd.read_csv(args.policies)
    for col in ("freq_params", "sev_params"):
        if col in policies.columns:
            policies[col] = policies[col].apply(_coerce_param_cell)

    sim = ClaimSimulator(
        policies_df=policies,
        random_seed=args.seed,
        correlation=args.correlation,
        copula_type=args.copula,
        copula_param=args.copula_param,
    )
    sim.simulate_claims()

    if args.nhpp:
        sim.simulate_dates_nhpp(
            lambda0=args.lambda0, alpha=args.alpha, phase=args.phase, T=args.T
        )

    claims = sim.claim_data
    if claims is None or claims.empty:
        print("No claims generated.")
        return 0

    print(f"Generated {len(claims)} claims across {claims['policy_id'].nunique()} policies.")
    loss_col = "ultimate_loss" if "ultimate_loss" in claims.columns else "amount"
    print(
        f"Total {loss_col}: {claims[loss_col].sum():,.2f}    "
        f"Mean: {claims[loss_col].mean():,.2f}    "
        f"Max: {claims[loss_col].max():,.2f}"
    )

    development_df = None
    if args.ldfs:
        if not args.nhpp:
            raise SystemExit(
                "--ldfs requires --nhpp so each claim has an incurred_date."
            )
        ldfs_raw = json.loads(Path(args.ldfs).read_text())
        base_ldfs = {int(k): float(v) for k, v in ldfs_raw.items()}
        sim.simulate_claim_development(
            base_LDFs=base_ldfs,
            volatility=args.dev_volatility,
            cumulative_factor=args.dev_cumulative_factor,
        )
        development_df = sim.claim_development
        print(
            f"Generated {len(development_df)} development records across "
            f"{development_df['accident_year'].nunique()} accident years."
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        claims.to_csv(out_path, index=False)
        print(f"Wrote claims to {out_path}")
        if development_df is not None:
            dev_path = out_path.with_name(out_path.stem + "_development.csv")
            development_df.to_csv(dev_path, index=False)
            print(f"Wrote claim development to {dev_path}")
    else:
        print()
        print(claims.head(10).to_string(index=False))
    return 0


def cmd_develop(args: argparse.Namespace) -> int:
    """Build a claim development triangle from an existing claims CSV."""
    from . import ClaimSimulator

    df = pd.read_csv(args.input)
    required = {"event_id", "policy_id", "ultimate_loss", "incurred_date"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Input is missing required columns: {missing}. "
            f"Expected: {sorted(required)}."
        )

    # ClaimSimulator.simulate_claim_development iterates over self.claim_data
    # and requires a policies_df with the schema below. Build a stub so we can
    # reuse the package implementation without re-coding the LDF logic here.
    stub_policies = pd.DataFrame(
        [
            {
                "policy_id": pid,
                "freq_dist": "poisson",
                "freq_params": (1,),
                "sev_dist": "normal",
                "sev_params": (0.0, 1.0),
                "start_date": "2000-01-01",
                "end_date": "2000-12-31",
            }
            for pid in df["policy_id"].dropna().unique()
        ]
    )
    sim = ClaimSimulator(policies_df=stub_policies, random_seed=args.seed)
    sim.claim_data = df.copy()
    sim.claim_data["incurred_date"] = pd.to_datetime(sim.claim_data["incurred_date"])

    ldfs_raw = json.loads(Path(args.ldfs).read_text())
    base_ldfs = {int(k): float(v) for k, v in ldfs_raw.items()}
    sim.simulate_claim_development(
        base_LDFs=base_ldfs,
        volatility=args.volatility,
        cumulative_factor=args.cumulative_factor,
    )

    dev = sim.claim_development
    print(
        f"Generated {len(dev)} development rows across "
        f"{dev['accident_year'].nunique()} accident years and "
        f"{dev['development_month'].nunique()} development months."
    )

    if args.triangle:
        tri = dev.pivot_table(
            index="accident_year",
            columns="development_month",
            values="incurred_loss",
            aggfunc="sum",
        ).round(2)
        print()
        print("Cumulative incurred-loss triangle:")
        print(tri.to_string())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dev.to_csv(out_path, index=False)
        print(f"Wrote claim development to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actsim",
        description="Actuarial risk modeling and simulation toolkit.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the package version and exit."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # version
    p = sub.add_parser("version", help="Print the package version.")
    p.set_defaults(func=cmd_version)

    # config
    p_cfg = sub.add_parser("config", help="Inspect package configuration.")
    cfg_sub = p_cfg.add_subparsers(dest="config_command", metavar="<subcommand>")
    p_show = cfg_sub.add_parser("show", help="Show the active config as JSON.")
    p_show.add_argument("--config", help="Optional path to a custom YAML config.")
    p_show.set_defaults(func=cmd_config_show)
    p_path = cfg_sub.add_parser("path", help="Print the path to the default config.yaml.")
    p_path.set_defaults(func=cmd_config_path)

    # fit
    p_fit = sub.add_parser(
        "fit", help="Fit candidate distributions to a univariate dataset."
    )
    p_fit.add_argument("--input", required=True, help="Path to CSV or JSON data file.")
    p_fit.add_argument("--column", help="Column to fit (required if file has >1 columns).")
    p_fit.add_argument(
        "--distributions", nargs="+",
        help="Distributions to try (overrides config).",
    )
    p_fit.add_argument(
        "--metrics", nargs="+",
        help=(
            f"Metrics to compute (default all: {', '.join(ALL_FIT_METRICS)}). "
            "Used to drive best-fit selection."
        ),
    )
    p_fit.add_argument(
        "--metric", default="aic", choices=ALL_FIT_METRICS,
        help="Metric used to pick the best fit (default: aic).",
    )
    p_fit.add_argument("--config", help="Custom YAML config path.")
    p_fit.add_argument("--output", help="Optional CSV path for the summary table.")
    # Truncation / data cleaning passthroughs to DistributionFitter.truncate_data
    p_fit.add_argument(
        "--truncate-remove",
        help="Comma-separated values to drop before fitting (e.g. '0,-999').",
    )
    p_fit.add_argument("--truncate-lower", type=float, help="Drop values below this.")
    p_fit.add_argument("--truncate-upper", type=float, help="Drop values above this.")
    p_fit.add_argument("--truncate-q-low", type=float, help="Drop below this quantile.")
    p_fit.add_argument("--truncate-q-high", type=float, help="Drop above this quantile.")
    p_fit.set_defaults(func=cmd_fit)

    # sample
    p_smp = sub.add_parser(
        "sample",
        help="Fit then draw random samples from the best (or chosen) distribution.",
    )
    p_smp.add_argument("--input", required=True, help="Path to CSV or JSON data file.")
    p_smp.add_argument("--column", help="Column to fit on.")
    p_smp.add_argument("--distributions", nargs="+",
                       help="Candidate distributions (defaults to config).")
    p_smp.add_argument(
        "--distribution",
        help="Force sampling from this specific distribution name.",
    )
    p_smp.add_argument("--size", type=int, default=1000, help="Number of samples.")
    p_smp.add_argument(
        "--zero-prop", type=float, default=0.0,
        help="Proportion of zero values to mix in (sample_mixed).",
    )
    p_smp.add_argument(
        "--one-prop", type=float, default=0.0,
        help="Proportion of one values to mix in (sample_mixed).",
    )
    p_smp.add_argument("--output", help="Optional CSV path for the samples.")
    p_smp.set_defaults(func=cmd_sample)

    # simulate
    p_sim = sub.add_parser(
        "simulate", help="Run an aggregate frequency/severity Monte Carlo simulation."
    )
    p_sim.add_argument("--freq-dist", required=True, help="Frequency distribution name.")
    p_sim.add_argument(
        "--freq-params", required=True,
        help="Comma-separated frequency params, e.g. '5'.",
    )
    p_sim.add_argument("--sev-dist", required=True, help="Severity distribution name.")
    p_sim.add_argument(
        "--sev-params", required=True,
        help="Comma-separated severity params, e.g. '7,1.2'.",
    )
    p_sim.add_argument("--num-sim", type=int, default=10000, help="Number of simulations.")
    p_sim.add_argument("--seed", type=int, default=1, help="Random seed.")
    p_sim.add_argument("--correlation", type=float, help="Frequency/severity correlation.")
    p_sim.add_argument(
        "--copula", choices=["gaussian", "frank", "gumbel", "clayton"],
        help="Copula type for dependence.",
    )
    p_sim.add_argument("--theta", type=float, default=0, help="Copula theta parameter.")
    p_sim.add_argument(
        "--keep-all", action="store_true",
        help="Keep per-event data (needed for OEP, layer treatment, event-level export).",
    )
    p_sim.add_argument(
        "--quantiles", type=float, nargs="+",
        help="Quantiles to report (default 0.5 0.75 0.9 0.95 0.99).",
    )
    # Layer treatment (requires --keep-all)
    p_sim.add_argument("--per-occ-ded", type=float, help="Per-occurrence deductible.")
    p_sim.add_argument("--per-occ-limit", type=float, help="Per-occurrence limit.")
    p_sim.add_argument("--agg-ded", type=float, help="Annual aggregate deductible.")
    p_sim.add_argument("--agg-limit", type=float, help="Annual aggregate limit.")
    p_sim.add_argument("--output", help="Optional CSV path for simulation output.")
    p_sim.set_defaults(func=cmd_simulate)

    # simulate-corr
    p_corr = sub.add_parser(
        "simulate-corr",
        help="Run a multivariate correlated simulation across lines of business.",
    )
    p_corr.add_argument("--corr-matrix", required=True, help="CSV correlation matrix.")
    p_corr.add_argument("--dist-list", required=True, help="JSON list of LoB distributions.")
    p_corr.add_argument("--num-sim", type=int, default=10000)
    p_corr.add_argument("--seed", type=int, default=1)
    p_corr.add_argument(
        "--keep-all", action="store_true",
        help="Export marginal simulations alongside the aggregate.",
    )
    p_corr.add_argument("--quantiles", type=float, nargs="+")
    p_corr.add_argument("--output", help="Optional CSV path for simulation output.")
    p_corr.set_defaults(func=cmd_simulate_corr)

    # claims
    p_cl = sub.add_parser(
        "claims", help="Simulate claims from a policy book defined in CSV."
    )
    p_cl.add_argument("--policies", required=True, help="Policies CSV file.")
    p_cl.add_argument("--seed", type=int, default=42)
    p_cl.add_argument("--correlation", type=float)
    p_cl.add_argument("--copula", choices=["gaussian", "frank", "gumbel", "clayton"])
    p_cl.add_argument("--copula-param", type=float, default=0)
    p_cl.add_argument(
        "--nhpp", action="store_true",
        help="Assign incurred dates using a non-homogeneous Poisson process.",
    )
    p_cl.add_argument("--lambda0", type=float, default=10)
    p_cl.add_argument("--alpha", type=float, default=0.5)
    p_cl.add_argument("--phase", type=float, default=0)
    p_cl.add_argument("--T", type=float, default=1)
    # Optional inline development triangle generation
    p_cl.add_argument(
        "--ldfs",
        help="JSON file with base loss development factors keyed by development month.",
    )
    p_cl.add_argument(
        "--dev-volatility", type=float, default=0.1,
        help="LDF volatility for per-claim noise (default 0.1).",
    )
    p_cl.add_argument(
        "--dev-cumulative-factor", type=float, default=1.0,
        help="Tail cumulative factor for development (default 1.0).",
    )
    p_cl.add_argument("--output", help="Optional CSV path for simulated claims.")
    p_cl.set_defaults(func=cmd_claims)

    # develop
    p_dev = sub.add_parser(
        "develop",
        help="Build a claim development triangle from an existing claims CSV.",
    )
    p_dev.add_argument(
        "--input", required=True,
        help="Claims CSV with columns: event_id, policy_id, ultimate_loss, incurred_date.",
    )
    p_dev.add_argument(
        "--ldfs", required=True,
        help="JSON file with base LDFs keyed by development month, e.g. {\"12\": 2.5, \"24\": 1.5}.",
    )
    p_dev.add_argument("--volatility", type=float, default=0.1)
    p_dev.add_argument("--cumulative-factor", type=float, default=1.0)
    p_dev.add_argument("--seed", type=int, default=42)
    p_dev.add_argument(
        "--triangle", action="store_true",
        help="Pivot the result into an accident-year x development-month triangle.",
    )
    p_dev.add_argument("--output", help="Optional CSV path for the development data.")
    p_dev.set_defaults(func=cmd_develop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    if args.command == "config" and not getattr(args, "config_command", None):
        args.func = cmd_config_show
        args.config = None

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
