from pprint import pprint

from incident_env import HedgingIncidentEnv


MARKET = {
    "S0": 100.0,
    "K": 100.0,
    "T": 1.0,
    "sigma_true": 0.20,
}


def show_case(
    title,
    env,
):
    print(
        "\n"
        + "=" * 90
    )
    print(title)
    print(
        "=" * 90
    )

    print(
        "\n[T0 initial observation]"
    )
    pprint(
        env.initial_observation()
    )

    print(
        "\n[Market history: step 10~35, stride 5]"
    )
    pprint(
        env.get_market_history(
            start_step=10,
            end_step=35,
            stride=5,
        )
    )

    print(
        "\n[Model decision history: step 10~35]"
    )
    pprint(
        env.get_model_decision_history(
            start_step=10,
            end_step=35,
        )
    )

    print(
        "\n[Order submissions: step 10~35]"
    )
    pprint(
        env.get_order_history(
            start_step=10,
            end_step=35,
        )
    )

    print(
        "\n[Fills: step 10~45]"
    )
    pprint(
        env.get_fill_history(
            start_step=10,
            end_step=45,
        )
    )


stale = HedgingIncidentEnv(
    incident_type=
        "stale_observation",
    seed=11,
    n_paths=3000,
    fault_params={
        "observation_lag_steps": 5,
    },
    market_params=MARKET,
    audit_path_index=0,
)

execution = HedgingIncidentEnv(
    incident_type=
        "execution_delay",
    seed=11,
    n_paths=3000,
    fault_params={
        "execution_lag_steps": 5,
    },
    market_params=MARKET,
    audit_path_index=0,
)

show_case(
    "STALE OBSERVATION",
    stale,
)

show_case(
    "EXECUTION DELAY",
    execution,
)
