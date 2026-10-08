from pprint import pprint
from incident_env import HedgingIncidentEnv


def main():
    env = HedgingIncidentEnv(
        incident_type="stale_observation",
        seed=11,
        n_paths=3000,
    )

    print("\n=== 1. Initial observation ===")
    pprint(env.initial_observation())

    print("\n=== 2. Market evidence ===")
    pprint(env.inspect_market())

    print("\n=== 3. Delta trajectory ===")
    pprint(env.inspect_delta_trajectory())

    print("\n=== 4. Observation alignment ===")
    pprint(env.inspect_observation_alignment())

    print("\n=== 5. Counterfactual: remove observation delay ===")
    pprint(env.run_counterfactual("remove_observation_delay"))

    print("\n=== 6. Evaluator-only ground truth ===")
    pprint(env.evaluator_ground_truth())


if __name__ == "__main__":
    main()
