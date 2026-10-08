import os
import numpy as np
import pandas as pd

from incident_env import HedgingIncidentEnv


INCIDENTS = [
    "normal",
    "vol_misspec",
    "under_rebalance",
    "stale_observation",
    "execution_delay",
]

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

N_PATHS = 3000


def collect_one_episode(incident, seed):
    env = HedgingIncidentEnv(
        incident_type=incident,
        seed=seed,
        n_paths=N_PATHS,
    )

    initial = env.initial_observation()
    delta_info = env.inspect_delta_trajectory()
    obs_info = env.inspect_observation_alignment()
    trade_info = env.inspect_trade_activity()

    if hasattr(env, "inspect_execution_timing"):
        execution_info = env.inspect_execution_timing()
    else:
        execution_info = {
            "decision_event_count": np.nan,
            "execution_event_count": np.nan,
            "max_pending_orders": np.nan,
        }

    return {
        "incident": incident,
        "seed": seed,
        "mean_pnl": initial["mean_pnl"],
        "rmse_pnl": initial["rmse_pnl"],
        "cvar95_pnl": initial["cvar95_pnl"],
        "mean_abs_delta_gap": delta_info["mean_abs_delta_gap"],
        "max_abs_delta_gap": delta_info["max_abs_delta_gap"],
        "mean_abs_price_observation_gap":
            obs_info["mean_abs_price_observation_gap"],
        "max_abs_price_observation_gap":
            obs_info["max_abs_price_observation_gap"],
        "mean_transaction_cost":
            trade_info["mean_transaction_cost"],
        "mean_turnover_dollars":
            trade_info["mean_turnover_dollars"],
        "rebalance_event_count":
            trade_info["rebalance_event_count"],
        "decision_event_count":
            execution_info["decision_event_count"],
        "execution_event_count":
            execution_info["execution_event_count"],
        "max_pending_orders":
            execution_info["max_pending_orders"],
    }


def main():
    rows = []

    print("=" * 90)
    print("MULTI-SEED RCA PILOT")
    print("=" * 90)

    total_runs = len(INCIDENTS) * len(SEEDS)
    run_idx = 0

    for incident in INCIDENTS:
        print(f"\n### {incident}")

        for seed in SEEDS:
            run_idx += 1
            row = collect_one_episode(
                incident=incident,
                seed=seed,
            )
            rows.append(row)

            print(
                f"[{run_idx:02d}/{total_runs}] "
                f"seed={seed:<3d} "
                f"RMSE={row['rmse_pnl']:.4f} "
                f"CVaR={row['cvar95_pnl']:.4f} "
                f"DeltaGap={row['mean_abs_delta_gap']:.4f}"
            )

    df = pd.DataFrame(rows)

    os.makedirs("outputs", exist_ok=True)

    raw_path = "outputs/multi_seed_results.csv"
    summary_path = "outputs/multi_seed_summary.csv"

    df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    numeric_cols = [
        c for c in df.columns
        if c not in ["incident", "seed"]
    ]

    summary = (
        df.groupby("incident")[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    summary.to_csv(
        summary_path,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 90)
    print("MEAN ± STD BY INCIDENT")
    print("=" * 90)

    core_metrics = [
        "rmse_pnl",
        "cvar95_pnl",
        "mean_abs_delta_gap",
        "mean_transaction_cost",
        "mean_turnover_dollars",
    ]

    for incident in INCIDENTS:
        part = df[df["incident"] == incident]

        print(f"\n[{incident}]")
        for metric in core_metrics:
            mean = part[metric].mean()
            std = part[metric].std()

            print(
                f"{metric:<30s} "
                f"{mean:>9.4f} ± {std:.4f}"
            )

    stale = df[df["incident"] == "stale_observation"]
    execution = df[df["incident"] == "execution_delay"]

    if len(stale) > 0 and len(execution) > 0:
        print("\n" + "=" * 90)
        print("CONFUSABLE PAIR CHECK")
        print("stale_observation vs execution_delay")
        print("=" * 90)

        for metric in [
            "rmse_pnl",
            "cvar95_pnl",
            "mean_abs_delta_gap",
            "mean_transaction_cost",
            "mean_turnover_dollars",
        ]:
            stale_mean = stale[metric].mean()
            exec_mean = execution[metric].mean()
            diff = abs(stale_mean - exec_mean)

            print(
                f"{metric:<30s} "
                f"stale={stale_mean:>9.4f} | "
                f"execution={exec_mean:>9.4f} | "
                f"|diff|={diff:.4f}"
            )

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
