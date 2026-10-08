import itertools
import os
import numpy as np
import pandas as pd

from incident_env import HedgingIncidentEnv

# pilot: 5 incidents x 9 regimes x 5 seeds = 225 runs
# full : severity까지 교차 = 765 runs
# MODE = "pilot"
MODE = "full"

K = 100.0
T = 1.0
N_PATHS = 3000

MONEYNESS_GRID = {
    "OTM": 0.90,
    "ATM": 1.00,
    "ITM": 1.10,
}

VOLATILITY_GRID = {
    "low": 0.15,
    "mid": 0.20,
    "high": 0.30,
}

SEEDS = [11, 22, 33, 44, 55]


# 이전 severity sweep에서 base regime에서 비교적 비슷했던 조합
PILOT_FAULTS = {
    "vol_misspec": {
        "severity_rank": 4,
        "fault_params": {"sigma_hedge_ratio": 0.60},
    },
    "under_rebalance": {
        "severity_rank": 3,
        "fault_params": {"rebalance_every": 20},
    },
    "stale_observation": {
        "severity_rank": 4,
        "fault_params": {"observation_lag_steps": 7},
    },
    "execution_delay": {
        "severity_rank": 4,
        "fault_params": {"execution_lag_steps": 7},
    },
}

FULL_SEVERITY_GRID = {
    "vol_misspec": [
        (1, {"sigma_hedge_ratio": 0.90}),
        (2, {"sigma_hedge_ratio": 0.80}),
        (3, {"sigma_hedge_ratio": 0.70}),
        (4, {"sigma_hedge_ratio": 0.60}),
    ],
    "under_rebalance": [
        (1, {"rebalance_every": 10}),
        (2, {"rebalance_every": 15}),
        (3, {"rebalance_every": 20}),
        (4, {"rebalance_every": 30}),
    ],
    "stale_observation": [
        (1, {"observation_lag_steps": 1}),
        (2, {"observation_lag_steps": 3}),
        (3, {"observation_lag_steps": 5}),
        (4, {"observation_lag_steps": 7}),
    ],
    "execution_delay": [
        (1, {"execution_lag_steps": 1}),
        (2, {"execution_lag_steps": 3}),
        (3, {"execution_lag_steps": 5}),
        (4, {"execution_lag_steps": 7}),
    ],
}


def build_cases():
    cases = [
        {
            "incident": "normal",
            "severity_rank": 0,
            "fault_params": None,
        }
    ]

    if MODE == "pilot":
        for incident, cfg in PILOT_FAULTS.items():
            cases.append({
                "incident": incident,
                "severity_rank": cfg["severity_rank"],
                "fault_params": cfg["fault_params"],
            })
    elif MODE == "full":
        for incident, levels in FULL_SEVERITY_GRID.items():
            for severity_rank, fault_params in levels:
                cases.append({
                    "incident": incident,
                    "severity_rank": severity_rank,
                    "fault_params": fault_params,
                })
    else:
        raise ValueError("MODE must be 'pilot' or 'full'")

    return cases


def fault_param_for_csv(fault_params):
    if not fault_params:
        return "none", np.nan
    name = next(iter(fault_params))
    return name, float(fault_params[name])


def collect_episode(
    incident,
    severity_rank,
    fault_params,
    moneyness_label,
    moneyness_ratio,
    vol_label,
    sigma_true,
    seed,
):
    S0 = K * moneyness_ratio

    env = HedgingIncidentEnv(
        incident_type=incident,
        seed=seed,
        n_paths=N_PATHS,
        fault_params=fault_params,
        market_params={
            "S0": S0,
            "K": K,
            "T": T,
            "sigma_true": sigma_true,
        },
    )

    initial = env.initial_observation()
    market = env.inspect_market()
    delta = env.inspect_delta_trajectory()
    obs = env.inspect_observation_alignment()
    trade = env.inspect_trade_activity()
    execution = env.inspect_execution_timing()

    param_name, param_value = fault_param_for_csv(fault_params)

    return {
        "incident": incident,
        "severity_rank": severity_rank,
        "fault_param_name": param_name,
        "fault_param_value": param_value,
        "seed": seed,

        "moneyness_label": moneyness_label,
        "moneyness_ratio": moneyness_ratio,
        "S0": S0,
        "K": K,
        "T": T,
        "volatility_regime": vol_label,
        "sigma_true": sigma_true,
        "realized_volatility_estimate":
            market["realized_volatility_estimate"],

        # T0 initial symptoms
        "mean_pnl": initial["mean_pnl"],
        "rmse_pnl": initial["rmse_pnl"],
        "cvar95_pnl": initial["cvar95_pnl"],

        # 분석용 diagnostic telemetry
        "mean_abs_delta_gap": delta["mean_abs_delta_gap"],
        "mean_abs_price_observation_gap":
            obs["mean_abs_price_observation_gap"],
        "mean_transaction_cost":
            trade["mean_transaction_cost"],
        "mean_turnover_dollars":
            trade["mean_turnover_dollars"],
        "rebalance_event_count":
            trade["rebalance_event_count"],
        "decision_event_count":
            execution["decision_event_count"],
        "execution_event_count":
            execution["execution_event_count"],
        "max_pending_orders":
            execution["max_pending_orders"],
    }


def make_summary(df):
    group_cols = [
        "incident",
        "severity_rank",
        "fault_param_name",
        "fault_param_value",
        "moneyness_label",
        "moneyness_ratio",
        "volatility_regime",
        "sigma_true",
    ]

    metrics = [
        "mean_pnl",
        "rmse_pnl",
        "cvar95_pnl",
        "mean_abs_delta_gap",
        "mean_abs_price_observation_gap",
        "mean_transaction_cost",
        "mean_turnover_dollars",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if str(x) != "")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def make_confusable_pairs(summary):
    """
    같은 market regime 안에서 서로 다른 root cause의
    T0(initial observation) 유사도를 계산.
    작을수록 초기 증상만으로 구별하기 어려움.
    """
    metrics = [
        "mean_pnl_mean",
        "rmse_pnl_mean",
        "cvar95_pnl_mean",
    ]

    scales = {}
    for metric in metrics:
        s = float(summary[metric].std())
        scales[metric] = s if np.isfinite(s) and s > 1e-12 else 1.0

    rows = []
    regime_cols = [
        "moneyness_label",
        "moneyness_ratio",
        "volatility_regime",
        "sigma_true",
    ]

    for regime_key, part in summary.groupby(regime_cols, dropna=False):
        records = part.to_dict("records")

        for a, b in itertools.combinations(records, 2):
            if a["incident"] == b["incident"]:
                continue

            distance_sq = 0.0
            for metric in metrics:
                z = (a[metric] - b[metric]) / scales[metric]
                distance_sq += z * z

            rows.append({
                "t0_distance": float(np.sqrt(distance_sq)),
                "moneyness_label": regime_key[0],
                "moneyness_ratio": regime_key[1],
                "volatility_regime": regime_key[2],
                "sigma_true": regime_key[3],
                "incident_a": a["incident"],
                "severity_a": a["severity_rank"],
                "incident_b": b["incident"],
                "severity_b": b["severity_rank"],
                "rmse_a": a["rmse_pnl_mean"],
                "rmse_b": b["rmse_pnl_mean"],
                "cvar_a": a["cvar95_pnl_mean"],
                "cvar_b": b["cvar95_pnl_mean"],
                "mean_pnl_a": a["mean_pnl_mean"],
                "mean_pnl_b": b["mean_pnl_mean"],
            })

    pairs = pd.DataFrame(rows)
    if len(pairs) == 0:
        return pairs

    return pairs.sort_values("t0_distance").reset_index(drop=True)


def main():
    cases = build_cases()
    regimes = list(itertools.product(
        MONEYNESS_GRID.items(),
        VOLATILITY_GRID.items(),
    ))

    total_runs = len(cases) * len(regimes) * len(SEEDS)
    rows = []
    run_idx = 0

    print("=" * 100)
    print("MARKET REGIME SWEEP")
    print(f"MODE={MODE} | total runs={total_runs}")
    print("=" * 100)

    for case in cases:
        for (
            (moneyness_label, moneyness_ratio),
            (vol_label, sigma_true),
        ) in regimes:

            print(
                f"\n### {case['incident']} "
                f"| S{case['severity_rank']} "
                f"| {moneyness_label} "
                f"| vol={vol_label}"
            )

            for seed in SEEDS:
                run_idx += 1
                row = collect_episode(
                    incident=case["incident"],
                    severity_rank=case["severity_rank"],
                    fault_params=case["fault_params"],
                    moneyness_label=moneyness_label,
                    moneyness_ratio=moneyness_ratio,
                    vol_label=vol_label,
                    sigma_true=sigma_true,
                    seed=seed,
                )
                rows.append(row)

                print(
                    f"[{run_idx:03d}/{total_runs}] "
                    f"seed={seed:<3d} "
                    f"RMSE={row['rmse_pnl']:.4f} "
                    f"CVaR={row['cvar95_pnl']:.4f}"
                )

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)

    raw_path = "outputs/market_regime_results.csv"
    summary_path = "outputs/market_regime_summary.csv"
    pairs_path = "outputs/market_regime_confusable_pairs.csv"
    normal_path = "outputs/normal_fault_overlap_candidates.csv"

    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = make_summary(df)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pairs = make_confusable_pairs(summary)
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    if len(pairs) > 0:
        normal_pairs = pairs[
            (pairs["incident_a"] == "normal")
            | (pairs["incident_b"] == "normal")
        ].copy()
        normal_pairs.to_csv(
            normal_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 100)
        print("TOP 20 CONFUSABLE PAIRS (same regime, T0 only)")
        print("=" * 100)

        cols = [
            "t0_distance",
            "moneyness_label",
            "volatility_regime",
            "incident_a",
            "severity_a",
            "incident_b",
            "severity_b",
            "rmse_a",
            "rmse_b",
            "cvar_a",
            "cvar_b",
        ]
        print(pairs[cols].head(20).to_string(index=False))

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {pairs_path}")
    print(f"  {normal_path}")


if __name__ == "__main__":
    main()
