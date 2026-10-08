from pprint import pprint

from rca_episode_env import RCAEpisodeEnv


env = RCAEpisodeEnv(
    incident_type="execution_delay",
    seed=1001,
    fault_params={
        "execution_lag_steps": 5
    },
    diagnosis_step=126,
)

print("[T0 at step 126]")
pprint(env.initial_observation())

print("\n[Request 120~200; result must stop at 126]")
pprint(
    env.get_market_history(
        120, 200
    )
)
