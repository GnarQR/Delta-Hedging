import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from build_episode_bank import make_env_from_row
from budgeted_rca_env import BudgetedRCAEnv
from systemone_client import SystemOneHTTPClient


load_dotenv()


LABELS = [
    "normal",
    "vol_misspec",
    "under_rebalance",
    "stale_observation",
    "execution_delay",
]

PROTOCOL_VERSION = "fixed_choice_v02"
PROMPT_VERSION = "fixed_choice_prompt_v02"


LABEL_CRITERIA = {
    "normal":
        "No injected hedging-system fault. Poor realized P&L alone is not sufficient evidence of a fault.",
    "vol_misspec":
        "The hedge model is using a materially inappropriate volatility input.",
    "under_rebalance":
        "Hedge decisions are being made less frequently than the intended hedge policy.",
    "stale_observation":
        "Delta decisions use delayed/stale market observations while execution itself is not the primary issue.",
    "execution_delay":
        "Delta decisions are made using current information, but hedge orders are filled later than intended.",
}


EVIDENCE_ACTION_CRITERIA = {
    "market_recent":
        "Inspect recent consecutive market prices; useful for checking whether a model-observed price matches a recent real market price.",
    "model_recent":
        "Inspect recent model decision records: decision step, observed stock price, target delta and volatility input.",
    "orders_recent":
        "Inspect recent hedge order submission records.",
    "fills_recent":
        "Inspect recent actual fill records.",
    "positions_recent":
        "Inspect recent hedge position and cash history.",
    "pnl_recent":
        "Inspect recent marked hedge P&L and cumulative transaction-cost history.",
    "realized_volatility":
        "Estimate realized volatility from observed market prices.",
    "diagnose_now":
        "Stop gathering evidence and make the provisional root-cause diagnosis now.",
}


COUNTERFACTUAL_CRITERIA = {
    "skip_counterfactual":
        "Do not run a counterfactual; current evidence is sufficient or no intervention is justified.",
    "use_realized_volatility":
        "Replay the same market path using observed realized volatility as the hedge volatility input.",
    "set_rebalance_every_5":
        "Replay the same market path with the intended five-step rebalancing interval.",
    "remove_observation_delay":
        "Replay the same market path with observation delay removed.",
    "remove_execution_delay":
        "Replay the same market path with execution delay removed.",
}


ACTION_CRITERIA = {
    "no_intervention":
        "Do not change the hedge system.",
    "use_realized_volatility":
        "Correct the volatility input using a market-based volatility estimate.",
    "set_rebalance_every_5":
        "Restore the intended five-step rebalancing frequency.",
    "remove_observation_delay":
        "Fix the market-data observation delay.",
    "remove_execution_delay":
        "Fix order execution latency.",
}


def normalize_probs(
    probabilities,
    keys,
):
    out = {
        k: float(
            probabilities.get(
                k,
                0.0,
            )
        )
        for k in keys
    }

    total = sum(
        max(v, 0.0)
        for v in out.values()
    )

    if total <= 0:
        return {
            k:
                1.0 / len(keys)
            for k in keys
        }

    return {
        k:
            max(v, 0.0)
            / total
        for k, v
        in out.items()
    }


def multiclass_brier(
    probabilities,
    truth,
):
    return float(
        sum(
            (
                probabilities[label]
                - (
                    1.0
                    if label == truth
                    else 0.0
                )
            ) ** 2
            for label in LABELS
        )
    )


def compact_query_record(
    record,
):
    """
    Compact records while preserving an explicit schema.

    v0.1 used anonymous arrays such as:
        [43, 215, 0.53]

    Small decision models were expected to infer the field meanings.
    v0.2 attaches schema and comparison guidance explicitly.
    """
    tool = record[
        "tool"
    ]
    result = record[
        "result"
    ]

    rows = result.get(
        "records"
    )

    schema = None
    guidance = None

    if tool == "get_market_history":
        schema = [
            "step",
            "stock_price",
        ]
        guidance = (
            "Compare stock_price at a decision step with the model's "
            "observed_stock_price at the SAME decision step."
        )

    elif tool == "get_model_decision_history":
        schema = [
            "decision_step",
            "observed_stock_price",
            "target_delta",
            "volatility_input",
        ]
        guidance = (
            "Decision-step spacing diagnoses under-rebalancing. "
            "Observed price versus same-step market price diagnoses stale input."
        )

    elif tool == "get_order_history":
        schema = [
            "order_id",
            "submitted_step",
            "target_position",
        ]
        guidance = (
            "Join orders to fills using order_id. Compare submitted_step "
            "with fill_step."
        )

    elif tool == "get_fill_history":
        schema = [
            "order_id",
            "fill_step",
            "fill_price",
            "resulting_position",
        ]
        guidance = (
            "Join fills to orders using order_id. A repeated positive "
            "fill_step - submitted_step gap supports execution delay."
        )

    elif tool == "get_position_history":
        schema = [
            "step",
            "position",
            "cash",
        ]

    elif tool == "get_pnl_history":
        schema = [
            "step",
            "marked_hedge_pnl",
            "cumulative_transaction_cost",
        ]

    if rows is None:
        return {
            "id":
                record[
                    "query_id"
                ],
            "tool":
                tool,
            "schema":
                {
                    "annualized_realized_volatility":
                        (
                            "volatility estimated from observed market returns; "
                            "a noisy market-based estimate"
                        )
                },
            "comparison_guidance":
                (
                    "Compare realized volatility with model volatility_input, "
                    "but do not treat either as exact ground truth."
                ),
            "result":
                result,
        }

    compact = []

    for row in rows:
        if tool == "get_market_history":
            compact.append(
                [
                    int(
                        row["step"]
                    ),
                    round(
                        float(
                            row[
                                "stock_price"
                            ]
                        ),
                        4,
                    ),
                ]
            )

        elif tool == "get_model_decision_history":
            compact.append(
                [
                    int(
                        row[
                            "decision_step"
                        ]
                    ),
                    round(
                        float(
                            row[
                                "observed_stock_price"
                            ]
                        ),
                        4,
                    ),
                    round(
                        float(
                            row[
                                "target_delta"
                            ]
                        ),
                        5,
                    ),
                    round(
                        float(
                            row[
                                "volatility_input"
                            ]
                        ),
                        4,
                    ),
                ]
            )

        elif tool == "get_order_history":
            compact.append(
                [
                    int(
                        row[
                            "order_id"
                        ]
                    ),
                    int(
                        row[
                            "submitted_step"
                        ]
                    ),
                    round(
                        float(
                            row[
                                "target_position"
                            ]
                        ),
                        5,
                    ),
                ]
            )

        elif tool == "get_fill_history":
            compact.append(
                [
                    int(
                        row[
                            "order_id"
                        ]
                    ),
                    int(
                        row[
                            "fill_step"
                        ]
                    ),
                    round(
                        float(
                            row[
                                "fill_price"
                            ]
                        ),
                        4,
                    ),
                    round(
                        float(
                            row[
                                "resulting_position"
                            ]
                        ),
                        5,
                    ),
                ]
            )

        elif tool == "get_position_history":
            compact.append(
                [
                    int(
                        row["step"]
                    ),
                    round(
                        float(
                            row[
                                "position"
                            ]
                        ),
                        5,
                    ),
                    round(
                        float(
                            row[
                                "cash"
                            ]
                        ),
                        4,
                    ),
                ]
            )

        elif tool == "get_pnl_history":
            compact.append(
                [
                    int(
                        row["step"]
                    ),
                    round(
                        float(
                            row[
                                "marked_hedge_pnl"
                            ]
                        ),
                        5,
                    ),
                    round(
                        float(
                            row[
                                "cumulative_transaction_cost"
                            ]
                        ),
                        5,
                    ),
                ]
            )

        else:
            compact.append(
                row
            )

    return {
        "id":
            record[
                "query_id"
            ],
        "tool":
            tool,
        "schema":
            schema,
        "comparison_guidance":
            guidance,
        "records":
            compact,
    }


def fixed_action_to_query(
    action,
    env,
    max_records=8,
):
    """
    Map a bounded evidence action to a deterministic, ALIGNED window.

    v0.2:
    market/model/order/fill evidence now refers to the same recent
    operational period so relational comparisons are possible.
    """
    t = int(
        env.diagnosis_step
    )

    expected_interval = int(
        env.base.healthy[
            "rebalance_every"
        ]
    )

    # Latest scheduled healthy decision step strictly before maturity/diagnosis.
    anchor = (
        (t // expected_interval)
        * expected_interval
    )

    if anchor >= t:
        anchor -= (
            expected_interval
        )

    anchor = max(
        anchor,
        0,
    )

    window_start = max(
        0,
        anchor
        - (
            max_records - 1
        )
        * expected_interval,
    )

    if action == "market_recent":
        return (
            "get_market_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    anchor,
                "stride":
                    expected_interval,
                "limit":
                    max_records,
            },
        )

    if action == "model_recent":
        return (
            "get_model_decision_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    anchor,
                "limit":
                    max_records,
            },
        )

    if action == "orders_recent":
        return (
            "get_order_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    anchor,
                "limit":
                    max_records,
            },
        )

    if action == "fills_recent":
        # Extend through diagnosis step so recently submitted orders
        # can be matched to delayed fills.
        return (
            "get_fill_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    t,
                "limit":
                    max_records,
            },
        )

    if action == "positions_recent":
        return (
            "get_position_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    anchor,
                "stride":
                    expected_interval,
                "limit":
                    max_records,
            },
        )

    if action == "pnl_recent":
        return (
            "get_pnl_history",
            {
                "start_step":
                    window_start,
                "end_step":
                    anchor,
                "stride":
                    expected_interval,
                "limit":
                    max_records,
            },
        )

    if action == "realized_volatility":
        return (
            "estimate_realized_volatility",
            {
                "start_step":
                    0,
                "end_step":
                    t,
            },
        )

    raise ValueError(
        f"Unknown fixed evidence action: {action}"
    )


def decision_state(
    initial_packet,
    compact_evidence,
    evidence_remaining,
    provisional=None,
    counterfactual=None,
):
    state = {
        "initial":
            initial_packet,
        "evidence":
            compact_evidence,
        "evidence_remaining":
            int(
                evidence_remaining
            ),
        "interpretation_rules": {
            "stale_observation":
                (
                    "Compare market stock_price and model observed_stock_price "
                    "at the SAME decision_step."
                ),
            "execution_delay":
                (
                    "Join order and fill records by order_id, then compare "
                    "submitted_step with fill_step against the expected delay."
                ),
            "under_rebalance":
                (
                    "Compare spacing between decision_step values with "
                    "expected rebalance_every_steps."
                ),
            "vol_misspec":
                (
                    "Compare model volatility_input with market-based realized "
                    "volatility. Realized volatility is noisy supporting evidence."
                ),
            "counterfactual":
                (
                    "Single-path counterfactual P&L is secondary and stochastic. "
                    "Do not overturn stronger audit-log mechanism evidence solely "
                    "because realized P&L failed to improve."
                ),
        },
    }

    if provisional is not None:
        state[
            "provisional_diagnosis"
        ] = provisional

    if counterfactual is not None:
        state[
            "counterfactual_result"
        ] = counterfactual

    return state


class GeminiChoiceClient:
    """
    Gemini adapter for the SAME bounded-choice protocol used by
    Jev/Laya/OpenJev.

    This is useful for apples-to-apples comparison:
    same state, same candidate options, same evidence menu.
    """

    provider_name = "gemini"

    def __init__(
        self,
        model=None,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Install google-genai first."
            ) from exc

        key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = (
            genai.Client(
                api_key=key
            )
        )

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL"
            )
            or "gemini-2.5-flash-lite"
        )

    def choose(
        self,
        *,
        state,
        question_id,
        instructions,
        criteria,
    ):
        prompt = {
            "task":
                instructions,
            "state":
                state,
            "allowed_choices":
                criteria,
            "output_schema": {
                "choice":
                    "one key from allowed_choices",
                "probabilities":
                    "object containing every allowed choice; values sum to 1",
            },
            "rule":
                "Return one JSON object only. No markdown and no explanation.",
        }

        response = (
            self.client.models.generate_content(
                model=self.model,
                contents=json.dumps(
                    prompt,
                    ensure_ascii=False,
                ),
            )
        )

        raw_text = (
            response.text
            or ""
        ).strip()

        # tolerate fenced output
        if raw_text.startswith(
            "```"
        ):
            raw_text = (
                raw_text
                .replace(
                    "```json",
                    "",
                    1,
                )
                .replace(
                    "```",
                    "",
                )
                .strip()
            )

        first = raw_text.find(
            "{"
        )
        last = raw_text.rfind(
            "}"
        )

        if (
            first >= 0
            and last > first
        ):
            raw_text = (
                raw_text[
                    first:
                    last + 1
                ]
            )

        payload = json.loads(
            raw_text
        )

        choice = str(
            payload[
                "choice"
            ]
        )

        if choice not in criteria:
            raise RuntimeError(
                f"Gemini returned invalid choice: {choice}"
            )

        probs = normalize_probs(
            payload.get(
                "probabilities",
                {},
            ),
            list(
                criteria.keys()
            ),
        )

        return {
            "provider":
                "gemini",
            "resolved_model":
                self.model,
            "choice":
                choice,
            "probabilities":
                probs,
            "confidence":
                max(
                    probs.values()
                ),
            "usage":
                None,
            "raw":
                payload,
        }


def make_choice_client(
    provider,
    model=None,
):
    provider = str(
        provider
    ).lower()

    if provider == "gemini":
        return GeminiChoiceClient(
            model=model
        )

    if provider in {
        "jev",
        "laya",
        "openjev",
    }:
        client = (
            SystemOneHTTPClient.from_provider(
                provider
            )
        )

        if model:
            client.model = model

        return client

    raise ValueError(
        f"Unsupported decision provider: {provider}"
    )


def decision_state(
    initial_packet,
    compact_evidence,
    evidence_remaining,
    provisional=None,
    counterfactual=None,
):
    state = {
        "initial":
            initial_packet,
        "evidence":
            compact_evidence,
        "evidence_remaining":
            int(
                evidence_remaining
            ),
    }

    if provisional is not None:
        state[
            "provisional_diagnosis"
        ] = provisional

    if counterfactual is not None:
        state[
            "counterfactual_result"
        ] = counterfactual

    return state


def run_one_episode(
    env,
    client,
    evidence_budget=4,
    max_records=8,
    counterfactual_budget=1,
):
    b = BudgetedRCAEnv(
        env,
        evidence_budget=
            evidence_budget,
        max_records_per_query=
            max_records,
        counterfactual_budget=
            counterfactual_budget,
    )

    initial = (
        b.initial_packet()
    )

    compact_evidence = []
    action_trace = []

    # --------------------------------------------------------
    # Active evidence acquisition
    # --------------------------------------------------------
    while (
        b.evidence_remaining
        > 0
    ):
        state = decision_state(
            initial,
            compact_evidence,
            b.evidence_remaining,
        )

        allowed = dict(
            EVIDENCE_ACTION_CRITERIA
        )

        # Don't allow exactly the same evidence family twice.
        used_actions = {
            x[
                "action"
            ]
            for x in action_trace
            if x[
                "stage"
            ]
            == "evidence"
        }

        for action in list(
            allowed.keys()
        ):
            if (
                action
                in used_actions
                and action
                != "diagnose_now"
            ):
                allowed.pop(
                    action
                )

        answer = client.choose(
            state=state,
            question_id=
                "next_action",
            instructions=(
                "Choose the single next action that most efficiently "
                "distinguishes the remaining root-cause hypotheses."
            ),
            criteria=allowed,
        )

        action = answer[
            "choice"
        ]

        action_trace.append(
            {
                "stage":
                    "evidence",
                "action":
                    action,
                "probabilities":
                    answer[
                        "probabilities"
                    ],
            }
        )

        if action == "diagnose_now":
            break

        tool, args = (
            fixed_action_to_query(
                action,
                env,
                max_records=
                    max_records,
            )
        )

        query = b.query(
            tool,
            **args,
        )

        compact_evidence.append(
            compact_query_record(
                query
            )
        )

    # --------------------------------------------------------
    # Provisional diagnosis
    # --------------------------------------------------------
    state = decision_state(
        initial,
        compact_evidence,
        b.evidence_remaining,
    )

    provisional_answer = (
        client.choose(
            state=state,
            question_id=
                "root_cause",
            instructions=(
                "Choose the most likely root cause from the audit evidence. "
                "Use the explicit schemas and relational comparison rules. "
                "Use normal only when evidence does not support an injected fault."
            ),
            criteria=
                LABEL_CRITERIA,
        )
    )

    provisional_probs = (
        normalize_probs(
            provisional_answer[
                "probabilities"
            ],
            LABELS,
        )
    )

    provisional = {
        "primary_label":
            provisional_answer[
                "choice"
            ],
        "probabilities":
            provisional_probs,
    }

    b.open_counterfactual_stage(
        provisional
    )

    # --------------------------------------------------------
    # One bounded counterfactual choice
    # --------------------------------------------------------
    cf_record = None

    if (
        counterfactual_budget
        > 0
    ):
        state = decision_state(
            initial,
            compact_evidence,
            b.evidence_remaining,
            provisional=
                provisional,
        )

        cf_answer = (
            client.choose(
                state=state,
                question_id=
                    "counterfactual_action",
                instructions=(
                    "Choose at most one causal replay that best targets the "
                    "provisional mechanism. Remember that single-path P&L is noisy."
                ),
                criteria=
                    COUNTERFACTUAL_CRITERIA,
            )
        )

        cf_action = (
            cf_answer[
                "choice"
            ]
        )

        action_trace.append(
            {
                "stage":
                    "counterfactual_choice",
                "action":
                    cf_action,
                "probabilities":
                    cf_answer[
                        "probabilities"
                    ],
            }
        )

        if (
            cf_action
            != "skip_counterfactual"
        ):
            if (
                cf_action
                == "set_rebalance_every_5"
            ):
                cf_record = (
                    b.run_counterfactual(
                        "set_rebalance_every",
                        5,
                    )
                )
            else:
                cf_record = (
                    b.run_counterfactual(
                        cf_action
                    )
                )

    # --------------------------------------------------------
    # Final diagnosis + recommended action
    # --------------------------------------------------------
    state = decision_state(
        initial,
        compact_evidence,
        b.evidence_remaining,
        provisional=
            provisional,
        counterfactual=
            cf_record,
    )

    final_answer = (
        client.choose(
            state=state,
            question_id=
                "final_root_cause",
            instructions=(
                "Choose the final root-cause diagnosis. Give direct audit-log "
                "mechanism evidence priority over noisy single-path P&L. "
                "Do not reject a hypothesis solely because its counterfactual "
                "did not improve realized terminal P&L."
            ),
            criteria=
                LABEL_CRITERIA,
        )
    )

    final_probs = (
        normalize_probs(
            final_answer[
                "probabilities"
            ],
            LABELS,
        )
    )

    action_answer = (
        client.choose(
            state=state,
            question_id=
                "recommended_action",
            instructions=(
                "Choose the most appropriate corrective action. "
                "If the episode is healthy, choose no_intervention."
            ),
            criteria=
                ACTION_CRITERIA,
        )
    )

    truth = (
        env.evaluator_ground_truth()[
            "incident_type"
        ]
    )

    return {
        "protocol_version":
            PROTOCOL_VERSION,
        "prompt_version":
            PROMPT_VERSION,
        "episode_id":
            env.episode_id,
        "provider":
            getattr(
                client,
                "provider_name",
                getattr(
                    client,
                    "provider",
                    "unknown",
                ),
            ),
        "model":
            getattr(
                client,
                "model",
                None,
            ),
        "truth":
            truth,
        "provisional":
            provisional,
        "final": {
            "primary_label":
                final_answer[
                    "choice"
                ],
            "probabilities":
                final_probs,
        },
        "recommended_action":
            action_answer[
                "choice"
            ],
        "provisional_correct":
            int(
                provisional[
                    "primary_label"
                ]
                == truth
            ),
        "final_correct":
            int(
                final_answer[
                    "choice"
                ]
                == truth
            ),
        "final_changed_from_provisional":
            int(
                final_answer[
                    "choice"
                ]
                != provisional[
                    "primary_label"
                ]
            ),
        "counterfactual_helped":
            int(
                provisional[
                    "primary_label"
                ]
                != truth
                and final_answer[
                    "choice"
                ]
                == truth
            ),
        "counterfactual_harmed":
            int(
                provisional[
                    "primary_label"
                ]
                == truth
                and final_answer[
                    "choice"
                ]
                != truth
            ),
        "final_brier":
            multiclass_brier(
                final_probs,
                truth,
            ),
        "evidence_used":
            b.evidence_used,
        "counterfactual_used":
            b.counterfactual_used,
        "evidence":
            compact_evidence,
        "counterfactual":
            cf_record,
        "action_trace":
            action_trace,
    }


def flat_row(
    result,
):
    row = {
        "episode_id":
            result[
                "episode_id"
            ],
        "provider":
            result[
                "provider"
            ],
        "model":
            result.get(
                "model"
            ),
        "truth":
            result[
                "truth"
            ],
        "provisional_label":
            result[
                "provisional"
            ][
                "primary_label"
            ],
        "final_label":
            result[
                "final"
            ][
                "primary_label"
            ],
        "provisional_correct":
            result[
                "provisional_correct"
            ],
        "final_correct":
            result[
                "final_correct"
            ],
        "final_brier":
            result[
                "final_brier"
            ],
        "evidence_used":
            result[
                "evidence_used"
            ],
        "counterfactual_used":
            result[
                "counterfactual_used"
            ],
        "final_changed_from_provisional":
            result[
                "final_changed_from_provisional"
            ],
        "counterfactual_helped":
            result[
                "counterfactual_helped"
            ],
        "counterfactual_harmed":
            result[
                "counterfactual_harmed"
            ],
        "recommended_action":
            result[
                "recommended_action"
            ],
    }

    for label in LABELS:
        row[
            f"p_{label}"
        ] = (
            result[
                "final"
            ][
                "probabilities"
            ][
                label
            ]
        )

    return row


class ScriptedDecisionClient:
    """
    API-free smoke-test client.
    """
    provider_name = "scripted"
    model = "scripted-decision-smoke"

    def __init__(self):
        self.calls = 0

    def choose(
        self,
        *,
        state,
        question_id,
        instructions,
        criteria,
    ):
        self.calls += 1

        if question_id == "next_action":
            preferred = [
                "orders_recent",
                "fills_recent",
                "diagnose_now",
            ]
            choice = next(
                (
                    x
                    for x in preferred
                    if x in criteria
                ),
                list(
                    criteria.keys()
                )[0],
            )

        elif question_id in {
            "root_cause",
            "final_root_cause",
        }:
            choice = (
                "execution_delay"
                if "execution_delay"
                in criteria
                else list(
                    criteria.keys()
                )[0]
            )

        elif question_id == "counterfactual_action":
            choice = (
                "remove_execution_delay"
                if "remove_execution_delay"
                in criteria
                else "skip_counterfactual"
            )

        elif question_id == "recommended_action":
            choice = (
                "remove_execution_delay"
                if "remove_execution_delay"
                in criteria
                else "no_intervention"
            )

        else:
            choice = list(
                criteria.keys()
            )[0]

        probs = {
            key:
                (
                    0.85
                    if key == choice
                    else (
                        0.15
                        / max(
                            len(criteria)
                            - 1,
                            1,
                        )
                    )
                )
            for key
            in criteria
        }

        if len(criteria) == 1:
            probs[
                choice
            ] = 1.0

        return {
            "provider":
                self.provider_name,
            "resolved_model":
                self.model,
            "choice":
                choice,
            "probabilities":
                probs,
            "confidence":
                max(
                    probs.values()
                ),
            "usage":
                None,
            "raw":
                None,
        }


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--provider",
        default="gemini",
        choices=[
            "gemini",
            "jev",
            "laya",
            "openjev",
            "smoke",
        ],
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--manifest",
        default=(
            "results/"
            "episode_bank_v01/"
            "episode_bank_balanced.csv"
        ),
    )

    parser.add_argument(
        "--split",
        default="dev",
        choices=[
            "train",
            "dev",
            "test",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--counterfactual-budget",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "decision_rca_run"
        ),
    )

    args = (
        parser.parse_args()
    )

    bank = pd.read_csv(
        args.manifest
    )

    part = (
        bank[
            bank[
                "split"
            ]
            == args.split
        ]
        .sort_values(
            "episode_id"
        )
        .reset_index(
            drop=True
        )
    )

    if args.limit is not None:
        part = part.head(
            args.limit
        )

    if args.provider == "smoke":
        # pick one execution-delay episode so scripted prediction is coherent
        execution = part[
            part[
                "incident_type"
            ]
            == "execution_delay"
        ]

        if len(execution):
            part = execution.head(
                1
            )

        client = (
            ScriptedDecisionClient()
        )

    else:
        client = (
            make_choice_client(
                args.provider,
                model=
                    args.model,
            )
        )

    output = Path(
        args.output_dir
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        output
        / "decision_episode_results.jsonl"
    )

    rows = []

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as f:
        for i, (_, row) in enumerate(
            part.iterrows(),
            start=1,
        ):
            env = (
                make_env_from_row(
                    row
                )
            )

            result = (
                run_one_episode(
                    env,
                    client,
                    evidence_budget=
                        args.evidence_budget,
                    max_records=
                        args.max_records,
                    counterfactual_budget=
                        args.counterfactual_budget,
                )
            )

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            rows.append(
                flat_row(
                    result
                )
            )

            print(
                f"[{i:03d}/{len(part):03d}] "
                f"{env.episode_id} "
                f"truth={result['truth']} "
                f"pred={result['final']['primary_label']} "
                f"correct={result['final_correct']} "
                f"q={result['evidence_used']}"
            )

    frame = pd.DataFrame(
        rows
    )

    csv_path = (
        output
        / "decision_episode_results.csv"
    )

    frame.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "provider":
            args.provider,
        "model":
            getattr(
                client,
                "model",
                args.model,
            ),
        "protocol":
            PROTOCOL_VERSION,
        "prompt_version":
            PROMPT_VERSION,
        "split":
            args.split,
        "n_episodes":
            int(
                len(frame)
            ),
        "provisional_accuracy":
            (
                float(
                    frame[
                        "provisional_correct"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "final_accuracy":
            (
                float(
                    frame[
                        "final_correct"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "accuracy":
            (
                float(
                    frame[
                        "final_correct"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "diagnosis_changed_rate":
            (
                float(
                    frame[
                        "final_changed_from_provisional"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "counterfactual_help_rate":
            (
                float(
                    frame[
                        "counterfactual_helped"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "counterfactual_harm_rate":
            (
                float(
                    frame[
                        "counterfactual_harmed"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "mean_brier":
            (
                float(
                    frame[
                        "final_brier"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "mean_evidence_queries":
            (
                float(
                    frame[
                        "evidence_used"
                    ].mean()
                )
                if len(frame)
                else None
            ),
        "evidence_budget":
            args.evidence_budget,
        "max_records":
            args.max_records,
        "counterfactual_budget":
            args.counterfactual_budget,
    }

    summary_path = (
        output
        / "decision_run_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
