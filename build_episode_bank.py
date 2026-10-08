import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rca_episode_env import RCAEpisodeEnv


# ============================================================
# Episode Bank v0.1
# ============================================================
#
# 핵심 원칙
# ---------
# 1. 하나의 market scenario(seed + regime)에 대해
#    여러 root cause / severity를 같은 시장경로에서 비교할 수 있게 한다.
#
# 2. train / dev / test는 base seed 자체를 분리한다.
#    따라서 동일 market shock realization이 split을 넘지 않는다.
#
# 3. Full bank는 모든 severity를 포함한다.
#
# 4. Balanced bank는 class별 episode 수를 동일하게 맞춘다.
#    normal은 severity가 하나뿐이므로 full bank는 자연스럽게
#    fault class가 더 많다. LLM 평가에는 balanced bank 사용을 권장.
#
# 5. 기본 BUILD_MODE="manifest"에서는 episode metadata만 만든다.
#    실제 raw audit log는 필요할 때 RCAEpisodeEnv로 재생성한다.
#
# 6. BUILD_MODE="materialized"로 바꾸면 모든 episode를 한 번 실행해서
#    T0 initial observation까지 bank에 캐시한다.
# ============================================================


BANK_VERSION = "v01"

# 빠른 기본값:
# - "manifest": metadata만 생성 + 일부 episode smoke validation
# - "materialized": 모든 episode를 실제 실행해서 T0까지 저장
BUILD_MODE = "manifest"

OUTPUT_DIR = Path("results") / f"episode_bank_{BANK_VERSION}"


# ------------------------------------------------------------
# Split
#
# 같은 base seed는 오직 하나의 split에만 존재.
#
# 각 base seed마다 9 market regimes가 있고,
# 각 regime마다 같은 random shocks를 공유하는 17개 episode가 생성:
#
#   normal 1
#   + 4 faults x 4 severities = 16
#   --------------------------------
#   17 episodes / scenario
#
# Full bank:
#   train 4 seeds -> 4 x 9 x 17 = 612
#   dev   2 seeds -> 2 x 9 x 17 = 306
#   test  4 seeds -> 4 x 9 x 17 = 612
#   total = 1530
#
# Balanced bank:
#   각 scenario에서 class별 1개씩 = 5 episodes
#   train = 180, dev = 90, test = 180
#   total = 450
# ------------------------------------------------------------
BASE_SEEDS_BY_SPLIT = {
    "train": [1001, 1002, 1003, 1004],
    "dev": [2001, 2002],
    "test": [3001, 3002, 3003, 3004],
}


# ------------------------------------------------------------
# Market regime grid
# ------------------------------------------------------------
K = 100.0
T = 1.0

MONEYNESS_GRID = [
    ("OTM", 0.90),
    ("ATM", 1.00),
    ("ITM", 1.10),
]

VOLATILITY_GRID = [
    ("low", 0.15),
    ("mid", 0.20),
    ("high", 0.30),
]


# ------------------------------------------------------------
# Fault severity grid
#
# vol_misspec:
#   absolute sigma가 아니라 sigma_true 대비 비율 사용.
#
# 나머지는 기존 severity sweep과 동일.
# ------------------------------------------------------------
FAULT_GRID = {
    "vol_misspec": [
        (1, "sigma_hedge_ratio", 0.90),
        (2, "sigma_hedge_ratio", 0.80),
        (3, "sigma_hedge_ratio", 0.70),
        (4, "sigma_hedge_ratio", 0.60),
    ],
    "under_rebalance": [
        (1, "rebalance_every", 10),
        (2, "rebalance_every", 15),
        (3, "rebalance_every", 20),
        (4, "rebalance_every", 30),
    ],
    "stale_observation": [
        (1, "observation_lag_steps", 1),
        (2, "observation_lag_steps", 3),
        (3, "observation_lag_steps", 5),
        (4, "observation_lag_steps", 7),
    ],
    "execution_delay": [
        (1, "execution_lag_steps", 1),
        (2, "execution_lag_steps", 3),
        (3, "execution_lag_steps", 5),
        (4, "execution_lag_steps", 7),
    ],
}

FAULT_ORDER = [
    "vol_misspec",
    "under_rebalance",
    "stale_observation",
    "execution_delay",
]


# ------------------------------------------------------------
# Diagnosis timing
#
# v0.1은 post-mortem만 사용.
# 향후 mid-episode 실험을 추가하고 싶으면
# 예: [126, 189, 252]처럼 확장 가능.
# ------------------------------------------------------------
DIAGNOSIS_STEPS = [252]


# manifest mode에서도 환경이 실제로 재생성되는지
# split마다 일부 episode를 smoke-test.
VALIDATION_EPISODES_PER_SPLIT = 5


def build_regimes():
    regimes = []

    regime_id = 0

    for m_label, m_ratio in MONEYNESS_GRID:
        for v_label, sigma_true in VOLATILITY_GRID:
            regimes.append(
                {
                    "regime_id": regime_id,
                    "moneyness_label": m_label,
                    "moneyness_ratio": float(m_ratio),
                    "volatility_regime": v_label,
                    "sigma_true": float(sigma_true),
                    "S0": float(K * m_ratio),
                    "K": float(K),
                    "T": float(T),
                }
            )
            regime_id += 1

    return regimes


def make_simulation_seed(
    base_seed,
    regime_id,
):
    """
    같은 base_seed + regime에서는 fault/severity가 달라도
    정확히 같은 market shocks를 사용.

    다른 regime은 다른 seed를 사용해,
    regime끼리 불필요하게 동일 shock path를 공유하지 않도록 함.
    """
    return int(
        base_seed * 100
        + regime_id
    )


def make_scenario_id(
    split,
    base_seed,
    regime_id,
):
    return (
        f"{split}"
        f"_b{base_seed}"
        f"_r{regime_id:02d}"
    )


def make_episode_id(
    scenario_id,
    incident_type,
    severity_rank,
    diagnosis_step,
):
    return (
        f"{scenario_id}"
        f"_{incident_type}"
        f"_s{severity_rank}"
        f"_t{diagnosis_step}"
    )


def build_manifest():
    rows = []
    regimes = build_regimes()

    for split, base_seeds in BASE_SEEDS_BY_SPLIT.items():
        for base_seed in base_seeds:
            for regime in regimes:
                regime_id = int(
                    regime["regime_id"]
                )

                simulation_seed = (
                    make_simulation_seed(
                        base_seed,
                        regime_id,
                    )
                )

                scenario_id = (
                    make_scenario_id(
                        split,
                        base_seed,
                        regime_id,
                    )
                )

                for diagnosis_step in DIAGNOSIS_STEPS:
                    # ----------------------------------------------
                    # Normal
                    # ----------------------------------------------
                    rows.append(
                        {
                            "bank_version":
                                BANK_VERSION,
                            "episode_id":
                                make_episode_id(
                                    scenario_id,
                                    "normal",
                                    0,
                                    diagnosis_step,
                                ),
                            "scenario_id":
                                scenario_id,
                            "split":
                                split,
                            "base_seed":
                                int(base_seed),
                            "simulation_seed":
                                int(simulation_seed),
                            "diagnosis_step":
                                int(diagnosis_step),

                            # Hidden evaluator labels
                            "incident_type":
                                "normal",
                            "severity_rank":
                                0,
                            "fault_param_name":
                                "none",
                            "fault_param_value":
                                np.nan,

                            # Market configuration
                            **regime,
                        }
                    )

                    # ----------------------------------------------
                    # Faults x severities
                    # ----------------------------------------------
                    for incident_type in FAULT_ORDER:
                        for (
                            severity_rank,
                            param_name,
                            param_value,
                        ) in FAULT_GRID[
                            incident_type
                        ]:
                            rows.append(
                                {
                                    "bank_version":
                                        BANK_VERSION,
                                    "episode_id":
                                        make_episode_id(
                                            scenario_id,
                                            incident_type,
                                            severity_rank,
                                            diagnosis_step,
                                        ),
                                    "scenario_id":
                                        scenario_id,
                                    "split":
                                        split,
                                    "base_seed":
                                        int(base_seed),
                                    "simulation_seed":
                                        int(
                                            simulation_seed
                                        ),
                                    "diagnosis_step":
                                        int(
                                            diagnosis_step
                                        ),

                                    # Hidden evaluator labels
                                    "incident_type":
                                        incident_type,
                                    "severity_rank":
                                        int(
                                            severity_rank
                                        ),
                                    "fault_param_name":
                                        param_name,
                                    "fault_param_value":
                                        float(
                                            param_value
                                        ),

                                    # Market configuration
                                    **regime,
                                }
                            )

    manifest = pd.DataFrame(rows)

    if manifest["episode_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest[
                "episode_id"
            ].duplicated(),
            "episode_id",
        ].tolist()

        raise RuntimeError(
            "Duplicate episode IDs: "
            f"{duplicates[:5]}"
        )

    return manifest


def row_to_fault_params(row):
    incident = str(
        row["incident_type"]
    )

    if incident == "normal":
        return None

    name = str(
        row["fault_param_name"]
    )
    value = float(
        row["fault_param_value"]
    )

    if name in {
        "rebalance_every",
        "observation_lag_steps",
        "execution_lag_steps",
    }:
        value = int(value)

    return {
        name: value
    }


def row_to_market_params(row):
    return {
        "S0":
            float(row["S0"]),
        "K":
            float(row["K"]),
        "T":
            float(row["T"]),
        "sigma_true":
            float(
                row["sigma_true"]
            ),
    }


def make_env_from_row(row):
    """
    나중에 LLM runner에서도 그대로 재사용 가능.

    예:
        row = manifest.iloc[0]
        env = make_env_from_row(row)
    """
    return RCAEpisodeEnv(
        incident_type=
            str(
                row[
                    "incident_type"
                ]
            ),
        seed=
            int(
                row[
                    "simulation_seed"
                ]
            ),
        fault_params=
            row_to_fault_params(
                row
            ),
        market_params=
            row_to_market_params(
                row
            ),
        diagnosis_step=
            int(
                row[
                    "diagnosis_step"
                ]
            ),
        episode_id=
            str(
                row[
                    "episode_id"
                ]
            ),
    )


def build_balanced_bank(
    manifest,
):
    """
    각 scenario마다:
        normal 1
        + fault class별 severity 1개
    를 선택.

    따라서 한 scenario당 5 classes가 정확히 1개씩 존재.

    severity는 scenario / fault에 따라 deterministic하게 순환해서
    전체 bank에서는 severity 1~4가 고르게 나오도록 함.
    """

    selected_ids = []

    fault_to_offset = {
        fault: i
        for i, fault
        in enumerate(
            FAULT_ORDER
        )
    }

    for scenario_id, part in manifest.groupby(
        "scenario_id",
        sort=True,
    ):
        # normal
        normal = part[
            part[
                "incident_type"
            ]
            == "normal"
        ]

        if len(normal) != len(
            DIAGNOSIS_STEPS
        ):
            raise RuntimeError(
                f"Unexpected normal rows in {scenario_id}"
            )

        for _, normal_row in normal.iterrows():
            selected_ids.append(
                normal_row[
                    "episode_id"
                ]
            )

        # fault classes
        base_seed = int(
            part.iloc[0][
                "base_seed"
            ]
        )
        regime_id = int(
            part.iloc[0][
                "regime_id"
            ]
        )

        for fault in FAULT_ORDER:
            fault_part = part[
                part[
                    "incident_type"
                ]
                == fault
            ]

            for diagnosis_step in DIAGNOSIS_STEPS:
                one_time = fault_part[
                    fault_part[
                        "diagnosis_step"
                    ]
                    == diagnosis_step
                ]

                # Deterministic severity rotation.
                severity_rank = (
                    (
                        base_seed
                        + regime_id
                        + fault_to_offset[
                            fault
                        ]
                    )
                    % 4
                ) + 1

                picked = one_time[
                    one_time[
                        "severity_rank"
                    ]
                    == severity_rank
                ]

                if len(picked) != 1:
                    raise RuntimeError(
                        "Could not select exactly one "
                        f"{fault} episode for {scenario_id}"
                    )

                selected_ids.append(
                    picked.iloc[0][
                        "episode_id"
                    ]
                )

    balanced = manifest[
        manifest[
            "episode_id"
        ].isin(
            selected_ids
        )
    ].copy()

    return (
        balanced.sort_values(
            [
                "split",
                "scenario_id",
                "incident_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def materialize_t0(
    manifest,
):
    """
    모든 episode를 실제 실행해 T0를 cache.

    raw market/order/fill history는 저장하지 않음.
    필요할 때 make_env_from_row()로 재생성.
    """
    rows = []

    total = len(manifest)

    print(
        f"\nMaterializing {total} episodes..."
    )

    for i, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        env = make_env_from_row(
            row
        )
        obs = (
            env.initial_observation()
        )

        out = row.to_dict()

        out.update(
            {
                "t0_marked_hedge_pnl":
                    obs[
                        "marked_hedge_pnl"
                    ],
                "t0_recent_20_step_pnl_change":
                    obs[
                        "recent_20_step_pnl_change"
                    ],
                "t0_worst_marked_hedge_pnl_to_date":
                    obs[
                        "worst_marked_hedge_pnl_to_date"
                    ],
                "t0_cumulative_transaction_cost":
                    obs[
                        "cumulative_transaction_cost"
                    ],
            }
        )

        rows.append(
            out
        )

        if (
            i == 1
            or i % 100 == 0
            or i == total
        ):
            print(
                f"[{i:04d}/{total}] "
                f"{row['episode_id']}"
            )

    return pd.DataFrame(
        rows
    )


def validate_sample_episodes(
    manifest,
):
    """
    manifest mode에서도 실제 env reconstruction이 되는지
    split마다 일부 episode를 실행해서 확인.
    """

    print(
        "\nValidating sample episodes..."
    )

    for split in [
        "train",
        "dev",
        "test",
    ]:
        part = manifest[
            manifest["split"]
            == split
        ]

        n = min(
            VALIDATION_EPISODES_PER_SPLIT,
            len(part),
        )

        # deterministic sample
        sample = part.sample(
            n=n,
            random_state=42,
        )

        for _, row in sample.iterrows():
            env = make_env_from_row(
                row
            )

            obs = (
                env.initial_observation()
            )

            if (
                obs["episode_id"]
                != row[
                    "episode_id"
                ]
            ):
                raise RuntimeError(
                    "Episode reconstruction mismatch"
                )

        print(
            f"  {split}: "
            f"{n} episodes passed"
        )


def make_summary(
    manifest,
):
    """
    사람이 bank 구성을 빠르게 확인하기 위한 요약.
    """

    summary = (
        manifest.groupby(
            [
                "split",
                "incident_type",
                "severity_rank",
                "moneyness_label",
                "volatility_regime",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="episode_count"
        )
    )

    return summary


def save_config():
    config = {
        "bank_version":
            BANK_VERSION,
        "build_mode":
            BUILD_MODE,
        "base_seeds_by_split":
            BASE_SEEDS_BY_SPLIT,
        "moneyness_grid":
            MONEYNESS_GRID,
        "volatility_grid":
            VOLATILITY_GRID,
        "K":
            K,
        "T":
            T,
        "fault_grid":
            {
                key: [
                    {
                        "severity_rank":
                            s,
                        "param_name":
                            name,
                        "param_value":
                            value,
                    }
                    for (
                        s,
                        name,
                        value,
                    )
                    in values
                ]
                for key, values
                in FAULT_GRID.items()
            },
        "diagnosis_steps":
            DIAGNOSIS_STEPS,
    }

    with open(
        OUTPUT_DIR
        / "episode_bank_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    if BUILD_MODE not in {
        "manifest",
        "materialized",
    }:
        raise ValueError(
            "BUILD_MODE must be "
            "'manifest' or 'materialized'"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        build_manifest()
    )

    # Split leakage sanity check
    split_seed_sets = {
        split:
            set(
                manifest.loc[
                    manifest[
                        "split"
                    ]
                    == split,
                    "base_seed",
                ]
            )
        for split
        in manifest[
            "split"
        ].unique()
    }

    splits = list(
        split_seed_sets
    )

    for i, a in enumerate(
        splits
    ):
        for b in splits[
            i + 1:
        ]:
            overlap = (
                split_seed_sets[a]
                & split_seed_sets[b]
            )

            if overlap:
                raise RuntimeError(
                    f"Seed leakage between "
                    f"{a} and {b}: "
                    f"{sorted(overlap)}"
                )

    if BUILD_MODE == "materialized":
        manifest = (
            materialize_t0(
                manifest
            )
        )
    else:
        validate_sample_episodes(
            manifest
        )

    balanced = (
        build_balanced_bank(
            manifest
        )
    )

    summary = (
        make_summary(
            manifest
        )
    )

    # Full evaluator-only manifest.
    manifest_path = (
        OUTPUT_DIR
        / "episode_bank_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Balanced evaluator manifest.
    balanced_path = (
        OUTPUT_DIR
        / "episode_bank_balanced.csv"
    )

    balanced.to_csv(
        balanced_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Agent runner에는 hidden label을 전달할 필요가 없음.
    # episode_id + split만 있는 안전한 index를 별도 저장.
    agent_index = balanced[
        [
            "episode_id",
            "split",
        ]
    ].copy()

    agent_index_path = (
        OUTPUT_DIR
        / "episode_bank_agent_index.csv"
    )

    agent_index.to_csv(
        agent_index_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = (
        OUTPUT_DIR
        / "episode_bank_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_config()

    print(
        "\n"
        + "=" * 90
    )
    print(
        "EPISODE BANK BUILT"
    )
    print(
        "=" * 90
    )

    print(
        f"\nFull bank: "
        f"{len(manifest)} episodes"
    )

    print(
        manifest.groupby(
            "split"
        ).size()
    )

    print(
        f"\nBalanced bank: "
        f"{len(balanced)} episodes"
    )

    print(
        balanced.groupby(
            [
                "split",
                "incident_type",
            ]
        ).size()
    )

    print(
        "\nUnique base seeds by split:"
    )

    for split, seeds in (
        split_seed_sets.items()
    ):
        print(
            f"  {split}: "
            f"{sorted(seeds)}"
        )

    print(
        "\nSaved:"
    )
    print(
        f"  {manifest_path}"
    )
    print(
        f"  {balanced_path}"
    )
    print(
        f"  {agent_index_path}"
    )
    print(
        f"  {summary_path}"
    )
    print(
        f"  "
        f"{OUTPUT_DIR / 'episode_bank_config.json'}"
    )


if __name__ == "__main__":
    main()
