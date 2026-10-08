from pprint import pprint

import pandas as pd

from build_episode_bank import (
    make_env_from_row,
)
from budgeted_rca_env import (
    BudgetedRCAEnv,
    BudgetExceededError,
)


BANK = (
    "results/"
    "episode_bank_v01/"
    "episode_bank_balanced.csv"
)

bank = pd.read_csv(
    BANK
)

# execution_delay episode 하나를 골라 budget policy를 눈으로 확인.
row = bank[
    bank[
        "incident_type"
    ]
    == "execution_delay"
].iloc[0]

env = make_env_from_row(
    row
)

b = BudgetedRCAEnv(
    env,
    evidence_budget=4,
    max_records_per_query=12,
    counterfactual_budget=1,
)

print(
    "\n[FREE INITIAL PACKET]"
)
pprint(
    b.initial_packet()
)

print(
    "\n[Q1 market]"
)
pprint(
    b.query(
        "get_market_history",
        start_step=0,
        end_step=252,
        stride=25,
        limit=999,  # server가 12로 강제
    )
)

print(
    "\n[Q2 model decisions]"
)
pprint(
    b.query(
        "get_model_decision_history",
        start_step=0,
        end_step=80,
        limit=999,
    )
)

print(
    "\n[Q3 orders]"
)
pprint(
    b.query(
        "get_order_history",
        start_step=0,
        end_step=80,
        limit=999,
    )
)

print(
    "\n[Q4 fills]"
)
pprint(
    b.query(
        "get_fill_history",
        start_step=0,
        end_step=90,
        limit=999,
    )
)

print(
    "\n[BUDGET STATUS]"
)
pprint(
    b.budget_status()
)

print(
    "\n[5th evidence query must fail]"
)

try:
    b.query(
        "get_pnl_history",
        start_step=0,
        end_step=20,
    )
except BudgetExceededError as exc:
    print(
        type(exc).__name__,
        str(exc),
    )

print(
    "\n[Open counterfactual stage]"
)

b.open_counterfactual_stage(
    {
        "primary_label":
            "execution_delay"
    }
)

print(
    "\n[CF1]"
)
pprint(
    b.run_counterfactual(
        "remove_execution_delay"
    )
)

print(
    "\n[FINAL BUDGET STATUS]"
)
pprint(
    b.budget_status()
)
