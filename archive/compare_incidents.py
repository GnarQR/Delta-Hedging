from pprint import pprint
from incident_env import HedgingIncidentEnv


INCIDENTS = [
    "normal",
    "vol_misspec",
    "under_rebalance",
    "stale_observation",
    "execution_delay",
]


def main():
    for incident in INCIDENTS:
        print("\n" + "=" * 80)
        print(f"INCIDENT: {incident}")
        print("=" * 80)

        env = HedgingIncidentEnv(
            incident_type=incident,
            seed=11,
            n_paths=3000,
        )

        print("\n[Initial observation]")
        pprint(env.initial_observation())

        print("\n[Delta trajectory]")
        pprint(env.inspect_delta_trajectory())

        print("\n[Observation alignment]")
        pprint(env.inspect_observation_alignment())

        print("\n[Execution timing]")
        pprint(env.inspect_execution_timing())

        print("\n[Trade activity]")
        pprint(env.inspect_trade_activity())

        print("\n[Evaluator-only truth]")
        pprint(env.evaluator_ground_truth())


if __name__ == "__main__":
    main()
