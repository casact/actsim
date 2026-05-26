"""Command-line interface for the actsim package."""
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


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


def _print_df(df: pd.DataFrame, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Wrote {len(df)} rows to {output}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(df.to_string(index=False))


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
    # _data is the parsed yaml dict
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
    metrics_default = cfg.metrics if cfg._data else None

    distributions = args.distributions or severity_default
    metrics = args.metrics or metrics_default or ["aic", "bic"]

    fitter = DistributionFitter(
        data=data, distributions=distributions, metrics=metrics
    )
    fitter.fit()

    summary = fitter.summary()[["name", *metrics]].sort_values(args.metric)
    print()
    print("Fit summary (sorted by {}):".format(args.metric))
    _print_df(summary, Path(args.output) if args.output else None)

    best = fitter.get_best_fit(args.metric)
    if best is not None:
        print()
        print(f"Best fit by {args.metric}: {best['name']}  params={best['params']}")
    return 0


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
    summary = {
        "VaR": [np.quantile(results_arr, q) for q in quantiles],
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

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.keep_all:
            sim.all_simulations.to_csv(out_path, index=False)
            print(f"Wrote per-event data to {out_path}")
        else:
            sim.results.to_frame("aggregate_loss").to_csv(out_path, index=False)
            print(f"Wrote aggregate simulations to {out_path}")
    return 0


def cmd_simulate_corr(args: argparse.Namespace) -> int:
    from . import StochasticSimulator

    # The multivariate routine reads dists/correlation from files, so we pass
    # placeholder marginals to satisfy the constructor.
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
        {"quantile": quantiles, "aggregate": [np.quantile(aggregate, q) for q in quantiles]}
    )
    print("Multivariate aggregate quantiles:")
    print(summary.to_string(index=False))
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

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        claims.to_csv(out_path, index=False)
        print(f"Wrote claims to {out_path}")
    else:
        print()
        print(claims.head(10).to_string(index=False))
    return 0


def _coerce_param_cell(value):
    """Best-effort coercion of a CSV cell into a tuple of numbers."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    s = str(value).strip()
    # Strip wrapping () or []
    if s.startswith(("(", "[")) and s.endswith((")", "]")):
        s = s[1:-1]
    return _parse_params(s)


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
        help="Goodness-of-fit metrics to compute (overrides config).",
    )
    p_fit.add_argument(
        "--metric", default="aic",
        help="Metric used to pick the best fit (default: aic).",
    )
    p_fit.add_argument("--config", help="Custom YAML config path.")
    p_fit.add_argument("--output", help="Optional CSV path for the summary table.")
    p_fit.set_defaults(func=cmd_fit)

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
        help="Keep per-event data (needed for OEP / event-level export).",
    )
    p_sim.add_argument(
        "--quantiles", type=float, nargs="+",
        help="Quantiles to report (default 0.5 0.75 0.9 0.95 0.99).",
    )
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
    p_cl.add_argument("--output", help="Optional CSV path for simulated claims.")
    p_cl.set_defaults(func=cmd_claims)

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
        # Default to "show" if user runs `actsim config`
        args.func = cmd_config_show
        args.config = None

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
