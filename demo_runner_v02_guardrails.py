import json
from pprint import pprint

import pandas as pd

from build_episode_bank import (
    make_env_from_row,
)
from budgeted_rca_env import (
    BudgetedRCAEnv,
    RepeatedEvidenceToolError,
)


BANK = (
    "results/"
    "episode_bank_v01/"
    "episode_bank_balanced.csv"
)

bank = pd.read_csv(
    BANK
)

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
    max_records_per_query=8,
    counterfactual_budget=1,
    max_queries_per_tool=1,
)

packet = b.initial_packet()

print(
    "\n[1] AGENT-VISIBLE INITIAL PACKET"
)
pprint(
    packet
)

serialized = json.dumps(
    packet
)

assert (
    "execution_delay"
    not in packet[
        "case_id"
    ]
)

assert (
    env.episode_id
    not in serialized
)

print(
    "\nPASS: hidden root-cause text is not exposed through episode_id."
)

print(
    "\n[2] FIRST MARKET QUERY"
)

q1 = b.query(
    "get_market_history",
    start_step=215,
    end_step=250,
    stride=5,
    limit=999,
)

pprint(
    q1
)

assert (
    "truncated"
    not in q1[
        "result"
    ]
)

assert (
    q1[
        "args"
    ][
        "limit"
    ]
    == 8
)

print(
    "\nPASS: record cap is enforced and raw `truncated` metadata is hidden."
)

print(
    "\n[3] REPEATED SAME-TOOL QUERY"
)

try:
    b.query(
        "get_market_history",
        start_step=0,
        end_step=10,
    )

    raise RuntimeError(
        "Repeat query unexpectedly succeeded"
    )

except RepeatedEvidenceToolError as exc:
    print(
        type(exc).__name__,
        str(exc),
    )

print(
    "\nPASS: pagination-style repeat is blocked."
)

print(
    "\n[4] EVALUATOR AUDIT LOG STILL RETAINS INTERFACE METADATA"
)

pprint(
    b.query_log
)

assert (
    "interface_metadata"
    in b.query_log[0]
)

print(
    "\nALL V0.2 GUARDRAILS PASSED"
)
