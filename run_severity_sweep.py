import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from incident_env import HedgingIncidentEnv


# ------------------------------------------------------------
# 1. Severity sweep 설정
#
# values는 반드시 "mild -> severe" 순서로 배치.
# severity_rank는 1,2,3,4로 자동 부여됨.
# ------------------------------------------------------------
SEVERITY_GRID = {
    "vol_misspec": {
        "param_name": "sigma_hedge",
        # true sigma=0.20이므로 값이 작아질수록 오류가 심해짐
        "values": [0.18, 0.16, 0.14, 0.12],
    },
    "under_rebalance": {
        "param_name": "rebalance_every",
        # 정상은 5 step, 값이 커질수록 업데이트가 드물어짐
        "values": [10, 15, 20, 30],
    },
    "stale_observation": {
        "param_name": "observation_lag_steps",
        "values": [1, 3, 5, 7],
    },
    "execution_delay": {
        "param_name": "execution_lag_steps",
        "values": [1, 3, 5, 7],
    },
}

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

N_PATHS = 3000


def collect_episode(
    incident,
    seed,
    severity_rank=0,
    param_name=None,
    param_value=None,
):
    """
    하나의 incident / severity / seed를 실행하고
    핵심 telemetry를 한 줄로 정리.
    """
    if incident == "normal":
        fault_params = None
    else:
        fault_params = {
            param_name: param_value
        }

    env = HedgingIncidentEnv(
        incident_type=incident,
        seed=seed,
        n_paths=N_PATHS,
        fault_params=fault_params,
    )

    initial = env.initial_observation()
    delta_info = env.inspect_delta_trajectory()
    obs_info = env.inspect_observation_alignment()
    trade_info = env.inspect_trade_activity()
    execution_info = env.inspect_execution_timing()

    return {
        "incident": incident,
        "severity_rank": severity_rank,
        "fault_param_name":
            "none" if param_name is None else param_name,
        "fault_param_value":
            np.nan if param_value is None else float(param_value),
        "seed": seed,

        # Initial symptom
        "mean_pnl": initial["mean_pnl"],
        "rmse_pnl": initial["rmse_pnl"],
        "cvar95_pnl": initial["cvar95_pnl"],

        # Delta mismatch
        "mean_abs_delta_gap":
            delta_info["mean_abs_delta_gap"],
        "max_abs_delta_gap":
            delta_info["max_abs_delta_gap"],

        # Observation timing
        "mean_abs_price_observation_gap":
            obs_info["mean_abs_price_observation_gap"],
        "max_abs_price_observation_gap":
            obs_info["max_abs_price_observation_gap"],

        # Trading
        "mean_transaction_cost":
            trade_info["mean_transaction_cost"],
        "mean_turnover_dollars":
            trade_info["mean_turnover_dollars"],
        "rebalance_event_count":
            trade_info["rebalance_event_count"],

        # Decision / execution timing
        "decision_event_count":
            execution_info["decision_event_count"],
        "execution_event_count":
            execution_info["execution_event_count"],
        "max_pending_orders":
            execution_info["max_pending_orders"],
    }


def make_summary(df):
    metrics = [
        "mean_pnl",
        "rmse_pnl",
        "cvar95_pnl",
        "mean_abs_delta_gap",
        "max_abs_delta_gap",
        "mean_abs_price_observation_gap",
        "max_abs_price_observation_gap",
        "mean_transaction_cost",
        "mean_turnover_dollars",
        "rebalance_event_count",
        "decision_event_count",
        "execution_event_count",
        "max_pending_orders",
    ]

    summary = (
        df.groupby(
            [
                "incident",
                "severity_rank",
                "fault_param_name",
                "fault_param_value",
            ],
            dropna=False,
        )[metrics]
        .agg(["mean", "std", "min", "max"])
        .round(5)
        .reset_index()
    )

    return summary


def make_flat_summary(df):
    """
    그래프와 matched-severity 후보 검색에 쓰기 편하도록
    mean/std만 flat column으로 만든 요약.
    """
    metrics = [
        "rmse_pnl",
        "cvar95_pnl",
        "mean_abs_delta_gap",
        "mean_transaction_cost",
        "mean_turnover_dollars",
    ]

    flat = (
        df.groupby(
            [
                "incident",
                "severity_rank",
                "fault_param_name",
                "fault_param_value",
            ],
            dropna=False,
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    flat.columns = [
        "_".join(
            str(x)
            for x in col
            if str(x) != ""
        )
        if isinstance(col, tuple)
        else col
        for col in flat.columns
    ]

    return flat


def find_matched_rmse_candidates(flat_summary):
    """
    네 fault에서 각각 severity 하나씩 고른 뒤,
    mean RMSE 범위(max-min)가 가장 작은 조합을 찾음.

    '정답 severity'를 자동 결정하는 함수가 아니라,
    우리가 사람이 검토할 후보를 좁히는 용도.
    """
    faults = [
        "vol_misspec",
        "under_rebalance",
        "stale_observation",
        "execution_delay",
    ]

    groups = {
        fault:
            flat_summary[
                flat_summary["incident"] == fault
            ].to_dict("records")
        for fault in faults
    }

    candidates = []

    for combo in product(
        groups["vol_misspec"],
        groups["under_rebalance"],
        groups["stale_observation"],
        groups["execution_delay"],
    ):
        rmse_means = [
            row["rmse_pnl_mean"]
            for row in combo
        ]

        cvar_means = [
            row["cvar95_pnl_mean"]
            for row in combo
        ]

        result = {
            "rmse_range":
                max(rmse_means) - min(rmse_means),
            "rmse_mean_across_faults":
                float(np.mean(rmse_means)),
            "cvar_range":
                max(cvar_means) - min(cvar_means),
        }

        for fault, row in zip(faults, combo):
            result[
                f"{fault}_severity_rank"
            ] = int(row["severity_rank"])

            result[
                f"{fault}_param_name"
            ] = row["fault_param_name"]

            result[
                f"{fault}_param_value"
            ] = row["fault_param_value"]

            result[
                f"{fault}_rmse_mean"
            ] = row["rmse_pnl_mean"]

            result[
                f"{fault}_cvar_mean"
            ] = row["cvar95_pnl_mean"]

        candidates.append(result)

    out = pd.DataFrame(candidates)

    return (
        out.sort_values(
            ["rmse_range", "cvar_range"]
        )
        .reset_index(drop=True)
    )


def plot_response_curves(flat_summary, output_dir):
    """
    severity_rank 기준 response curve.
    x축의 1,2,3,4는 각 fault 내부에서 mild -> severe 순서.
    서로 다른 fault의 raw parameter 단위가 다르므로
    x축을 raw value가 아니라 severity rank로 통일.
    """
    fault_df = flat_summary[
        flat_summary["incident"] != "normal"
    ].copy()

    # RMSE
    plt.figure(figsize=(9, 6))

    for incident, part in fault_df.groupby("incident"):
        part = part.sort_values("severity_rank")

        plt.errorbar(
            part["severity_rank"],
            part["rmse_pnl_mean"],
            yerr=part["rmse_pnl_std"],
            marker="o",
            capsize=3,
            label=incident,
        )

    plt.xlabel("Severity rank (1=mild, 4=severe)")
    plt.ylabel("RMSE of hedging P&L")
    plt.title("Fault severity vs hedging RMSE")
    plt.xticks([1, 2, 3, 4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "severity_rmse_curve.png",
        ),
        dpi=160,
    )
    plt.close()

    # CVaR
    plt.figure(figsize=(9, 6))

    for incident, part in fault_df.groupby("incident"):
        part = part.sort_values("severity_rank")

        plt.errorbar(
            part["severity_rank"],
            part["cvar95_pnl_mean"],
            yerr=part["cvar95_pnl_std"],
            marker="o",
            capsize=3,
            label=incident,
        )

    plt.xlabel("Severity rank (1=mild, 4=severe)")
    plt.ylabel("95% left-tail CVaR of hedging P&L")
    plt.title("Fault severity vs tail risk")
    plt.xticks([1, 2, 3, 4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "severity_cvar_curve.png",
        ),
        dpi=160,
    )
    plt.close()


def main():
    rows = []

    total_fault_runs = (
        sum(
            len(config["values"])
            for config in SEVERITY_GRID.values()
        )
        * len(SEEDS)
    )

    total_runs = total_fault_runs + len(SEEDS)
    run_index = 0

    print("=" * 100)
    print("SEVERITY SWEEP PILOT")
    print("=" * 100)

    # ------------------------------------------------------------
    # Normal은 severity가 없으므로 seed별로 한 번만 실행
    # ------------------------------------------------------------
    print("\n### normal")

    for seed in SEEDS:
        run_index += 1

        row = collect_episode(
            incident="normal",
            seed=seed,
        )
        rows.append(row)

        print(
            f"[{run_index:03d}/{total_runs}] "
            f"seed={seed:<3d} "
            f"RMSE={row['rmse_pnl']:.4f} "
            f"CVaR={row['cvar95_pnl']:.4f}"
        )

    # ------------------------------------------------------------
    # 각 fault × severity × seed 실행
    # ------------------------------------------------------------
    for incident, config in SEVERITY_GRID.items():
        param_name = config["param_name"]

        for severity_rank, param_value in enumerate(
            config["values"],
            start=1,
        ):
            print(
                f"\n### {incident} | "
                f"severity={severity_rank} | "
                f"{param_name}={param_value}"
            )

            for seed in SEEDS:
                run_index += 1

                row = collect_episode(
                    incident=incident,
                    seed=seed,
                    severity_rank=severity_rank,
                    param_name=param_name,
                    param_value=param_value,
                )
                rows.append(row)

                print(
                    f"[{run_index:03d}/{total_runs}] "
                    f"seed={seed:<3d} "
                    f"RMSE={row['rmse_pnl']:.4f} "
                    f"CVaR={row['cvar95_pnl']:.4f}"
                )

    df = pd.DataFrame(rows)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(
        output_dir,
        "severity_sweep_results.csv",
    )
    summary_path = os.path.join(
        output_dir,
        "severity_sweep_summary.csv",
    )
    flat_summary_path = os.path.join(
        output_dir,
        "severity_sweep_flat_summary.csv",
    )
    matched_path = os.path.join(
        output_dir,
        "severity_matched_candidates.csv",
    )

    df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = make_summary(df)
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    flat_summary = make_flat_summary(df)
    flat_summary.to_csv(
        flat_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    matched = find_matched_rmse_candidates(
        flat_summary
    )
    matched.head(20).to_csv(
        matched_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_response_curves(
        flat_summary,
        output_dir,
    )

    # ------------------------------------------------------------
    # 콘솔 요약
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("MEAN RMSE / CVaR BY FAULT SEVERITY")
    print("=" * 100)

    display = flat_summary[
        flat_summary["incident"] != "normal"
    ].copy()

    display = display.sort_values(
        ["incident", "severity_rank"]
    )

    for _, row in display.iterrows():
        print(
            f"{row['incident']:<20s} "
            f"S{int(row['severity_rank'])} "
            f"{row['fault_param_name']}="
            f"{row['fault_param_value']:<6g} | "
            f"RMSE={row['rmse_pnl_mean']:.4f}"
            f" ± {row['rmse_pnl_std']:.4f} | "
            f"CVaR={row['cvar95_pnl_mean']:.4f}"
            f" ± {row['cvar95_pnl_std']:.4f}"
        )

    print("\n" + "=" * 100)
    print("TOP 10 RMSE-MATCHED SEVERITY COMBINATIONS")
    print("=" * 100)

    columns_to_show = [
        "rmse_range",
        "rmse_mean_across_faults",
        "vol_misspec_param_value",
        "under_rebalance_param_value",
        "stale_observation_param_value",
        "execution_delay_param_value",
    ]

    print(
        matched[
            columns_to_show
        ].head(10).to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {flat_summary_path}")
    print(f"  {matched_path}")
    print(
        f"  {output_dir}/severity_rmse_curve.png"
    )
    print(
        f"  {output_dir}/severity_cvar_curve.png"
    )


if __name__ == "__main__":
    main()
