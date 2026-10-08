from pprint import pprint

from rca_episode_env import RCAEpisodeEnv


MARKET = {
    "S0": 100.0,
    "K": 100.0,
    "T": 1.0,
    "sigma_true": 0.20,
}


def show(title, env):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print("\n[T0]")
    pprint(env.initial_observation())

    print("\n[Market 15~30]")
    pprint(
        env.get_market_history(
            15, 30, stride=5
        )
    )

    print("\n[Model decisions 15~30]")
    pprint(
        env.get_model_decision_history(
            15, 30
        )
    )

    print("\n[Orders 15~30]")
    pprint(
        env.get_order_history(
            15, 30
        )
    )

    print("\n[Fills 15~35]")
    pprint(
        env.get_fill_history(
            15, 35
        )
    )

    print("\n[Final P&L window]")
    pprint(
        env.get_pnl_history(
            max(0, env.diagnosis_step - 20),
            env.diagnosis_step,
            stride=5,
        )
    )


stale = RCAEpisodeEnv(
    incident_type="stale_observation",
    seed=1001,
    fault_params={
        "observation_lag_steps": 5
    },
    market_params=MARKET,
)

execution = RCAEpisodeEnv(
    incident_type="execution_delay",
    seed=1001,
    fault_params={
        "execution_lag_steps": 5
    },
    market_params=MARKET,
)

show("STALE OBSERVATION EPISODE", stale)
show("EXECUTION DELAY EPISODE", execution)

print("\n" + "=" * 100)
print("COUNTERFACTUALS")
print("=" * 100)

print("\n[Stale -> remove observation delay]")
pprint(
    stale.run_counterfactual(
        "remove_observation_delay"
    )
)

print("\n[Execution -> remove execution delay]")
pprint(
    execution.run_counterfactual(
        "remove_execution_delay"
    )
)

print("\n[Evaluator-only causal score: never send this to the agent]")
pprint(
    execution.evaluator_counterfactual_effect(
        "remove_execution_delay"
    )
)

print("\n[Evaluator-only ground truth]")
pprint(
    execution.evaluator_ground_truth()
)
