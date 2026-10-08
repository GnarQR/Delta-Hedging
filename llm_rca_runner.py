import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load secrets/config from project-local .env if present.
load_dotenv()

from budgeted_rca_env import (
    BudgetedRCAEnv,
    BudgetExceededError,
    CounterfactualStageError,
    RepeatedEvidenceToolError,
)
from build_episode_bank import (
    make_env_from_row,
)


LABELS = [
    "normal",
    "vol_misspec",
    "under_rebalance",
    "stale_observation",
    "execution_delay",
]


PROTOCOL_VERSION = "flexible_agent_v02"
PROMPT_VERSION = "rca_prompt_v02"


SYSTEM_PROMPT = """
You are diagnosing one option delta-hedging incident.

All observations and evidence refer to the SAME single realized market path.

Closed-set scoring labels:
- normal
- vol_misspec
- under_rebalance
- stale_observation
- execution_delay

Do not assume poor realized P&L implies a fault.

Operational interpretation rules:
1. Stale observation:
   Compare a model decision's observed_stock_price with the actual market
   stock_price at the SAME decision step. A persistent mismatch supports
   stale_observation.

2. Execution delay:
   Match order and fill records by order_id, then compare submitted_step
   with fill_step. A persistent positive gap relative to the expected
   execution delay supports execution_delay.

3. Under-rebalancing:
   Compare spacing between model decision steps with the expected
   rebalance_every_steps in contract context.

4. Volatility misspecification:
   Compare the model's volatility_input with market-based realized-volatility
   evidence. Realized volatility is itself noisy, so use it as supporting
   rather than infallible evidence.

5. Interface record limits:
   Interface record caps or internal pagination are NOT evidence of a missing,
   stale, delayed, or corrupted market feed. Do not diagnose a fault because
   a query returned only the maximum allowed number of records.

Evidence-use rules:
- Select evidence that discriminates competing hypotheses.
- Do NOT spend multiple queries merely paginating the same log family.
- Each evidence tool is normally available once, so choose a targeted window.
- Never invent records that were not returned.

Counterfactual rule:
- Single-path counterfactual P&L is stochastic and SECONDARY evidence.
- A correct repair can make terminal P&L look worse on one realized path.
- Therefore, do not overturn strong audit-log evidence solely because
  counterfactual P&L failed to improve.
- Prefer direct mechanism evidence from timestamps, observed prices,
  decision spacing, and model inputs.

Output EXACTLY one JSON object and no markdown.
"""


def action_instructions(
    evidence_remaining,
    counterfactual_stage,
    counterfactual_remaining,
):
    labels_json = json.dumps(
        LABELS
    )

    if not counterfactual_stage:
        return f"""
Current stage: EVIDENCE ACQUISITION

Evidence queries remaining: {evidence_remaining}

Choose exactly one action.

A) Query evidence:
{{
  "action": "query",
  "tool": "<tool_name>",
  "args": {{ ... }}
}}

B) Stop gathering evidence and issue a provisional diagnosis:
{{
  "action": "provisional_diagnosis",
  "primary_label": "<one of {labels_json}>",
  "probabilities": {{
    "normal": 0.0,
    "vol_misspec": 0.0,
    "under_rebalance": 0.0,
    "stale_observation": 0.0,
    "execution_delay": 0.0
  }},
  "supporting_evidence": ["Q1"],
  "reason": "brief explanation"
}}

Probabilities must sum to 1.
"""

    if counterfactual_remaining > 0:
        return f"""
Current stage: COUNTERFACTUAL VERIFICATION

You already submitted a provisional diagnosis.

Choose exactly one action.

A) Run ONE counterfactual:
{{
  "action": "counterfactual",
  "intervention": "<one of:
      use_realized_volatility,
      set_volatility_input,
      set_rebalance_every,
      remove_observation_delay,
      remove_execution_delay>",
  "value": null
}}

Use a numeric value only when the chosen intervention requires it.

B) Skip the counterfactual and finalize:
{final_schema()}
"""

    return f"""
Current stage: FINAL DIAGNOSIS

The counterfactual budget is exhausted.
Return:
{final_schema()}
"""


def final_schema():
    labels_json = json.dumps(
        LABELS
    )

    return f"""
{{
  "action": "final",
  "primary_label": "<one of {labels_json}>",
  "probabilities": {{
    "normal": 0.0,
    "vol_misspec": 0.0,
    "under_rebalance": 0.0,
    "stale_observation": 0.0,
    "execution_delay": 0.0
  }},
  "supporting_evidence": ["Q1", "Q2", "CF1"],
  "recommended_action": "brief corrective action or no intervention",
  "reason": "brief final rationale"
}}
"""


def tool_help_text(
    budgeted_env,
):
    tools = (
        budgeted_env.available_evidence_tools()
    )

    schemas = {
        "get_market_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "stride": "int>=1",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "step": "market timestamp/index",
                "stock_price": "actual stock price at this step",
            },
            "comparison_hint":
                (
                    "For stale-observation diagnosis, compare stock_price "
                    "at a decision step with model observed_stock_price "
                    "at the SAME decision step."
                ),
        },
        "get_model_decision_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "decision_step": "step when hedge target was computed",
                "observed_stock_price": "stock price seen by the model",
                "target_delta": "computed target hedge position",
                "volatility_input": "volatility input used by the hedge model",
                "time_to_maturity": "remaining option maturity",
            },
            "comparison_hint":
                (
                    "Use decision-step spacing for under-rebalancing; "
                    "compare observed_stock_price with market stock_price "
                    "at the same step for stale observations."
                ),
        },
        "get_order_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "order_id": "relational key",
                "submitted_step": "step when order was submitted",
                "target_position": "target hedge position in that order",
            },
            "comparison_hint":
                (
                    "Join to fill history using order_id, then compare "
                    "submitted_step with fill_step."
                ),
        },
        "get_fill_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "order_id": "relational key matching order history",
                "fill_step": "step when the order actually filled",
                "fill_price": "stock price at fill",
                "shares_traded": "executed share change",
                "resulting_position": "position after fill",
                "transaction_cost": "cost of the fill",
            },
            "comparison_hint":
                (
                    "Join to order history using order_id. A repeated "
                    "fill_step > submitted_step relative to the expected "
                    "execution delay supports execution_delay."
                ),
        },
        "get_position_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "stride": "int>=1",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "step": "time step",
                "position": "actual stock hedge position",
                "cash": "cash account",
                "hedge_portfolio_value": "stock position plus cash",
            },
        },
        "get_pnl_history": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
                "stride": "int>=1",
                "limit": "int (server capped)",
            },
            "record_schema": {
                "step": "time step",
                "marked_hedge_pnl": "single-path marked hedge P&L",
                "cumulative_transaction_cost": "cost accumulated to that step",
            },
        },
        "estimate_realized_volatility": {
            "args": {
                "start_step": "int",
                "end_step": "int|null",
            },
            "record_schema": {
                "annualized_realized_volatility":
                    "volatility estimated from observed market returns",
            },
            "comparison_hint":
                (
                    "Compare with model volatility_input, but remember "
                    "realized volatility is a noisy estimate."
                ),
        },
    }

    return {
        name: {
            "description":
                description,
            **schemas[name],
        }
        for name, description
        in sorted(
            tools.items()
        )
    }


def parse_json_object(
    text,
):
    text = text.strip()

    if text.startswith(
        "```"
    ):
        lines = (
            text.splitlines()
        )

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    # Robust fallback:
    # extract first top-level-looking JSON object.
    if not text.startswith(
        "{"
    ):
        first = text.find(
            "{"
        )
        last = text.rfind(
            "}"
        )

        if (
            first >= 0
            and last > first
        ):
            text = text[
                first:
                last + 1
            ]

    return json.loads(
        text
    )


def validate_probabilities(
    payload,
):
    probs = payload.get(
        "probabilities"
    )

    if not isinstance(
        probs,
        dict,
    ):
        raise ValueError(
            "probabilities must be an object"
        )

    if set(
        probs.keys()
    ) != set(
        LABELS
    ):
        raise ValueError(
            "probabilities must contain exactly the five benchmark labels"
        )

    numeric = {}

    for label in LABELS:
        value = float(
            probs[
                label
            ]
        )

        if (
            not math.isfinite(
                value
            )
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(
                f"Invalid probability for {label}: {value}"
            )

        numeric[
            label
        ] = value

    total = sum(
        numeric.values()
    )

    if abs(
        total - 1.0
    ) > 0.02:
        raise ValueError(
            f"Probabilities sum to {total}, not 1"
        )

    # Tiny floating-point drift is normalized.
    if total > 0:
        numeric = {
            k: v / total
            for k, v
            in numeric.items()
        }

    primary = payload.get(
        "primary_label"
    )

    if primary not in LABELS:
        raise ValueError(
            f"Unknown primary_label: {primary}"
        )

    payload = dict(
        payload
    )
    payload[
        "probabilities"
    ] = numeric

    return payload


def multiclass_brier(
    probabilities,
    truth,
):
    """
    Multiclass Brier score:
        sum_k (p_k - y_k)^2

    Lower is better.
    """
    score = 0.0

    for label in LABELS:
        target = (
            1.0
            if label == truth
            else 0.0
        )

        score += (
            probabilities[
                label
            ]
            - target
        ) ** 2

    return float(
        score
    )



class OpenAIResponsesModel:
    """
    OpenAI adapter.

    The RCA tools remain local Python functions.
    The model only emits JSON actions.
    """

    provider = "openai"

    def __init__(
        self,
        model,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI Python package first: pip install -U openai"
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        self.client = OpenAI()
        self.model = str(model)

    def complete(
        self,
        messages,
    ):
        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        return response.output_text


class GeminiGenerateContentModel:
    """
    Gemini adapter using the official google-genai SDK.

    Environment variable:
        GEMINI_API_KEY

    Example model:
        gemini-2.5-flash-lite

    The runner keeps the evidence tools local and sends only
    conversation history to Gemini.
    """

    provider = "gemini"

    def __init__(
        self,
        model,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Install the Gemini SDK first: pip install -U google-genai"
            ) from exc

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )
        self.types = types
        self.model = str(model)

    def _convert_messages(
        self,
        messages,
    ):
        """
        Convert the runner's OpenAI-style message list into
        Gemini generate_content history.

        system messages -> system_instruction
        assistant       -> model role
        user            -> user role
        """
        system_parts = []
        contents = []

        for message in messages:
            role = str(
                message.get(
                    "role",
                    "user",
                )
            )

            content = message.get(
                "content",
                "",
            )

            if not isinstance(
                content,
                str,
            ):
                content = json.dumps(
                    content,
                    ensure_ascii=False,
                )

            if role == "system":
                system_parts.append(
                    content
                )
                continue

            gemini_role = (
                "model"
                if role == "assistant"
                else "user"
            )

            contents.append(
                self.types.Content(
                    role=gemini_role,
                    parts=[
                        self.types.Part.from_text(
                            text=content
                        )
                    ],
                )
            )

        system_instruction = (
            "\n\n".join(
                system_parts
            )
            if system_parts
            else None
        )

        return (
            system_instruction,
            contents,
        )

    def complete(
        self,
        messages,
    ):
        (
            system_instruction,
            contents,
        ) = self._convert_messages(
            messages
        )

        config = (
            self.types.GenerateContentConfig(
                system_instruction=
                    system_instruction,
            )
        )

        response = (
            self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned no text output."
            )

        return response.text


class ScriptedSmokeModel:
    """
    API 없이 runner state machine을 검증하기 위한 fake model.
    실제 실험에는 사용하지 않음.
    """

    def __init__(
        self,
    ):
        self.turn = 0

    def complete(
        self,
        messages,
    ):
        self.turn += 1

        scripted = {
            1: {
                "action": "query",
                "tool": "get_market_history",
                "args": {
                    "start_step": 10,
                    "end_step": 35,
                    "stride": 5,
                    "limit": 99,
                },
            },
            2: {
                "action": "query",
                "tool": "get_model_decision_history",
                "args": {
                    "start_step": 10,
                    "end_step": 35,
                    "limit": 99,
                },
            },
            3: {
                "action": "query",
                "tool": "get_order_history",
                "args": {
                    "start_step": 10,
                    "end_step": 35,
                    "limit": 99,
                },
            },
            4: {
                "action": "query",
                "tool": "get_fill_history",
                "args": {
                    "start_step": 10,
                    "end_step": 40,
                    "limit": 99,
                },
            },
            5: {
                "action": "provisional_diagnosis",
                "primary_label": "execution_delay",
                "probabilities": {
                    "normal": 0.02,
                    "vol_misspec": 0.02,
                    "under_rebalance": 0.03,
                    "stale_observation": 0.08,
                    "execution_delay": 0.85,
                },
                "supporting_evidence": [
                    "Q3",
                    "Q4",
                ],
                "reason": "submitted orders are filled later",
            },
            6: {
                "action": "counterfactual",
                "intervention": "remove_execution_delay",
                "value": None,
            },
            7: {
                "action": "final",
                "primary_label": "execution_delay",
                "probabilities": {
                    "normal": 0.01,
                    "vol_misspec": 0.01,
                    "under_rebalance": 0.02,
                    "stale_observation": 0.04,
                    "execution_delay": 0.92,
                },
                "supporting_evidence": [
                    "Q3",
                    "Q4",
                    "CF1",
                ],
                "recommended_action": "remove or investigate execution latency",
                "reason": "order submission and fill timing are misaligned",
            },
        }

        return json.dumps(
            scripted[
                self.turn
            ]
        )


class RCAAgentRunner:
    def __init__(
        self,
        model_client,
        evidence_budget=4,
        max_records_per_query=12,
        counterfactual_budget=1,
        max_queries_per_tool=1,
        max_model_turns=12,
    ):
        self.model_client = model_client
        self.evidence_budget = int(
            evidence_budget
        )
        self.max_records_per_query = int(
            max_records_per_query
        )
        self.counterfactual_budget = int(
            counterfactual_budget
        )
        self.max_queries_per_tool = int(
            max_queries_per_tool
        )
        self.max_model_turns = int(
            max_model_turns
        )

    def run_episode(
        self,
        env,
    ):
        b_env = BudgetedRCAEnv(
            env,
            evidence_budget=
                self.evidence_budget,
            max_records_per_query=
                self.max_records_per_query,
            counterfactual_budget=
                self.counterfactual_budget,
            max_queries_per_tool=
                self.max_queries_per_tool,
        )

        initial = (
            b_env.initial_packet()
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task":
                            "Diagnose the root cause of this hedging episode.",
                        "initial_packet":
                            initial,
                        "evidence_tools":
                            tool_help_text(
                                b_env
                            ),
                        "instructions":
                            action_instructions(
                                b_env.evidence_remaining,
                                False,
                                b_env.counterfactual_remaining,
                            ),
                    },
                    indent=2,
                ),
            },
        ]

        provisional = None
        final = None
        raw_turns = []

        for turn_index in range(
            1,
            self.max_model_turns + 1,
        ):
            raw = (
                self.model_client.complete(
                    messages
                )
            )

            raw_turns.append(
                raw
            )

            try:
                action = (
                    parse_json_object(
                        raw
                    )
                )
            except Exception as exc:
                messages.append(
                    {
                        "role": "assistant",
                        "content": raw,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "error":
                                    f"Invalid JSON: {exc}",
                                "instruction":
                                    "Return exactly one valid JSON object.",
                            }
                        ),
                    }
                )
                continue

            kind = action.get(
                "action"
            )

            # ----------------------------------------------------
            # Evidence stage
            # ----------------------------------------------------
            if not b_env.counterfactual_stage_open:
                if kind == "query":
                    tool = action.get(
                        "tool"
                    )

                    args = action.get(
                        "args",
                        {},
                    )

                    if not isinstance(
                        args,
                        dict,
                    ):
                        args = {}

                    try:
                        result = (
                            b_env.query(
                                tool,
                                **args,
                            )
                        )
                    except Exception as exc:
                        result = {
                            "error":
                                type(exc).__name__,
                            "message":
                                str(exc),
                            "evidence_remaining":
                                b_env.evidence_remaining,
                        }

                    messages.append(
                        {
                            "role": "assistant",
                            "content": raw,
                        }
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "tool_result":
                                        result,
                                    "instructions":
                                        action_instructions(
                                            b_env.evidence_remaining,
                                            False,
                                            b_env.counterfactual_remaining,
                                        ),
                                },
                                indent=2,
                            ),
                        }
                    )

                    continue

                if kind == "provisional_diagnosis":
                    try:
                        provisional = (
                            validate_probabilities(
                                action
                            )
                        )
                    except Exception as exc:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": raw,
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "error":
                                            str(exc),
                                        "instruction":
                                            "Return a valid provisional_diagnosis JSON.",
                                    }
                                ),
                            }
                        )
                        continue

                    b_env.open_counterfactual_stage(
                        provisional
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": raw,
                        }
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "status":
                                        "provisional diagnosis accepted",
                                    "instructions":
                                        action_instructions(
                                            b_env.evidence_remaining,
                                            True,
                                            b_env.counterfactual_remaining,
                                        ),
                                },
                                indent=2,
                            ),
                        }
                    )

                    continue

                # Invalid action for current stage.
                messages.append(
                    {
                        "role": "assistant",
                        "content": raw,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "error":
                                    "During evidence stage, action must be query or provisional_diagnosis.",
                                "instructions":
                                    action_instructions(
                                        b_env.evidence_remaining,
                                        False,
                                        b_env.counterfactual_remaining,
                                    ),
                            }
                        ),
                    }
                )

                continue

            # ----------------------------------------------------
            # Counterfactual / final stage
            # ----------------------------------------------------
            if kind == "counterfactual":
                intervention = (
                    action.get(
                        "intervention"
                    )
                )

                value = action.get(
                    "value"
                )

                try:
                    result = (
                        b_env.run_counterfactual(
                            intervention,
                            value,
                        )
                    )
                except Exception as exc:
                    result = {
                        "error":
                            type(exc).__name__,
                        "message":
                            str(exc),
                        "counterfactual_remaining":
                            b_env.counterfactual_remaining,
                    }

                messages.append(
                    {
                        "role": "assistant",
                        "content": raw,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "counterfactual_result":
                                    result,
                                "instructions":
                                    action_instructions(
                                        b_env.evidence_remaining,
                                        True,
                                        b_env.counterfactual_remaining,
                                    ),
                            },
                            indent=2,
                        ),
                    }
                )

                continue

            if kind == "final":
                try:
                    final = (
                        validate_probabilities(
                            action
                        )
                    )
                except Exception as exc:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": raw,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "error":
                                        str(exc),
                                    "instruction":
                                        "Return a valid final JSON.",
                                }
                            ),
                        }
                    )
                    continue

                break

            messages.append(
                {
                    "role": "assistant",
                    "content": raw,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "error":
                                "After provisional diagnosis, action must be counterfactual or final.",
                            "instructions":
                                action_instructions(
                                    b_env.evidence_remaining,
                                    True,
                                    b_env.counterfactual_remaining,
                                ),
                        }
                    ),
                }
            )

        if final is None:
            raise RuntimeError(
                "Model did not produce a valid final diagnosis within max_model_turns."
            )

        truth = (
            env.evaluator_ground_truth()[
                "incident_type"
            ]
        )

        provisional_correct = (
            None
            if provisional is None
            else int(
                provisional[
                    "primary_label"
                ]
                == truth
            )
        )

        final_correct = int(
            final[
                "primary_label"
            ]
            == truth
        )

        result = {
            "protocol_version":
                PROTOCOL_VERSION,
            "prompt_version":
                PROMPT_VERSION,
            "episode_id":
                env.episode_id,
            "truth":
                truth,
            "provisional":
                provisional,
            "final":
                final,
            "provisional_correct":
                provisional_correct,
            "final_correct":
                final_correct,
            "final_changed_from_provisional":
                (
                    None
                    if provisional is None
                    else int(
                        final[
                            "primary_label"
                        ]
                        != provisional[
                            "primary_label"
                        ]
                    )
                ),
            "counterfactual_helped":
                (
                    None
                    if provisional_correct is None
                    else int(
                        provisional_correct == 0
                        and final_correct == 1
                    )
                ),
            "counterfactual_harmed":
                (
                    None
                    if provisional_correct is None
                    else int(
                        provisional_correct == 1
                        and final_correct == 0
                    )
                ),
            "final_brier":
                multiclass_brier(
                    final[
                        "probabilities"
                    ],
                    truth,
                ),
            "evidence_used":
                b_env.evidence_used,
            "counterfactual_used":
                b_env.counterfactual_used,
            "query_log":
                b_env.query_log,
            "counterfactual_log":
                b_env.counterfactual_log,
            "raw_model_turns":
                raw_turns,
        }

        return result


def result_to_flat_row(
    result,
):
    final = result[
        "final"
    ]

    provisional = result.get(
        "provisional"
    )

    row = {
        "episode_id":
            result[
                "episode_id"
            ],
        "truth":
            result[
                "truth"
            ],
        "final_primary_label":
            final[
                "primary_label"
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
            result.get(
                "final_changed_from_provisional"
            ),
        "counterfactual_helped":
            result.get(
                "counterfactual_helped"
            ),
        "counterfactual_harmed":
            result.get(
                "counterfactual_harmed"
            ),
        "recommended_action":
            final.get(
                "recommended_action",
                "",
            ),
    }

    if provisional is not None:
        row[
            "provisional_primary_label"
        ] = provisional[
            "primary_label"
        ]
        row[
            "provisional_correct"
        ] = result[
            "provisional_correct"
        ]

    for label in LABELS:
        row[
            f"p_{label}"
        ] = final[
            "probabilities"
        ][
            label
        ]

    return row


def run_bank(
    manifest_path,
    split,
    model_client,
    output_dir,
    evidence_budget,
    max_records_per_query,
    counterfactual_budget,
    limit=None,
    offset=0,
):
    bank = pd.read_csv(
        manifest_path
    )

    part = (
        bank[
            bank[
                "split"
            ]
            == split
        ]
        .sort_values(
            "episode_id"
        )
        .reset_index(
            drop=True
        )
    )

    if offset:
        part = (
            part.iloc[
                int(offset):
            ]
            .reset_index(
                drop=True
            )
        )

    if limit is not None:
        part = (
            part.head(
                int(limit)
            )
        )

    runner = RCAAgentRunner(
        model_client,
        evidence_budget=
            evidence_budget,
        max_records_per_query=
            max_records_per_query,
        counterfactual_budget=
            counterfactual_budget,
    )

    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        output_dir
        / "episode_results.jsonl"
    )

    csv_path = (
        output_dir
        / "episode_results.csv"
    )

    flat_rows = []

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
                runner.run_episode(
                    env
                )
            )

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            flat_rows.append(
                result_to_flat_row(
                    result
                )
            )

            print(
                f"[{i:03d}/{len(part):03d}] "
                f"{env.episode_id} "
                f"truth={result['truth']} "
                f"pred={result['final']['primary_label']} "
                f"correct={result['final_correct']} "
                f"queries={result['evidence_used']}"
            )

    flat = pd.DataFrame(
        flat_rows
    )

    flat.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    if len(flat):
        final_accuracy = float(
            flat[
                "final_correct"
            ].mean()
        )

        if (
            "provisional_correct"
            in flat.columns
        ):
            provisional_accuracy = float(
                flat[
                    "provisional_correct"
                ].dropna().mean()
            )
        else:
            provisional_accuracy = float(
                "nan"
            )

        mean_brier = float(
            flat[
                "final_brier"
            ].mean()
        )

        mean_queries = float(
            flat[
                "evidence_used"
            ].mean()
        )

        changed_rate = float(
            flat[
                "final_changed_from_provisional"
            ].dropna().mean()
        )

        cf_help_rate = float(
            flat[
                "counterfactual_helped"
            ].dropna().mean()
        )

        cf_harm_rate = float(
            flat[
                "counterfactual_harmed"
            ].dropna().mean()
        )
    else:
        final_accuracy = float("nan")
        provisional_accuracy = float("nan")
        mean_brier = float("nan")
        mean_queries = float("nan")
        changed_rate = float("nan")
        cf_help_rate = float("nan")
        cf_harm_rate = float("nan")

    summary = {
        "provider":
            getattr(
                model_client,
                "provider",
                "unknown",
            ),
        "model":
            getattr(
                model_client,
                "model",
                "scripted-smoke",
            ),
        "protocol_version":
            PROTOCOL_VERSION,
        "prompt_version":
            PROMPT_VERSION,
        "split":
            split,
        "n_episodes":
            int(
                len(flat)
            ),
        "provisional_accuracy":
            provisional_accuracy,
        "final_accuracy":
            final_accuracy,
        "accuracy":
            final_accuracy,
        "diagnosis_changed_rate":
            changed_rate,
        "counterfactual_help_rate":
            cf_help_rate,
        "counterfactual_harm_rate":
            cf_harm_rate,
        "mean_brier":
            mean_brier,
        "mean_evidence_queries":
            mean_queries,
        "evidence_budget":
            int(
                evidence_budget
            ),
        "max_records_per_query":
            int(
                max_records_per_query
            ),
        "counterfactual_budget":
            int(
                counterfactual_budget
            ),
    }

    summary_path = (
        output_dir
        / "run_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nSummary:"
    )
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nSaved:"
    )
    print(
        jsonl_path
    )
    print(
        csv_path
    )
    print(
        summary_path
    )


def main():
    parser = argparse.ArgumentParser()

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
        "--provider",
        default=os.getenv(
            "LLM_PROVIDER",
            "gemini",
        ),
        choices=[
            "gemini",
            "openai",
        ],
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model name. "
            "Default for Gemini: gemini-2.5-flash-lite. "
            "For OpenAI, set --model or OPENAI_MODEL."
        ),
    )

    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--counterfactual-budget",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "llm_rca_run"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use a scripted fake model. "
            "No API call is made."
        ),
    )

    args = parser.parse_args()

    if args.smoke:
        model_client = (
            ScriptedSmokeModel()
        )

        # smoke model is hardcoded to one episode
        # and should only run one row.
        if args.limit is None:
            args.limit = 1

    else:
        if args.provider == "gemini":
            model_name = (
                args.model
                or os.getenv(
                    "GEMINI_MODEL"
                )
                or "gemini-2.5-flash-lite"
            )

            model_client = (
                GeminiGenerateContentModel(
                    model_name
                )
            )

        elif args.provider == "openai":
            model_name = (
                args.model
                or os.getenv(
                    "OPENAI_MODEL"
                )
            )

            if not model_name:
                raise RuntimeError(
                    "For provider=openai, pass --model "
                    "or set OPENAI_MODEL."
                )

            model_client = (
                OpenAIResponsesModel(
                    model_name
                )
            )

        else:
            raise RuntimeError(
                f"Unsupported provider: {args.provider}"
            )

    run_bank(
        manifest_path=
            args.manifest,
        split=
            args.split,
        model_client=
            model_client,
        output_dir=
            args.output_dir,
        evidence_budget=
            args.evidence_budget,
        max_records_per_query=
            args.max_records,
        counterfactual_budget=
            args.counterfactual_budget,
        limit=
            args.limit,
        offset=
            args.offset,
    )


if __name__ == "__main__":
    main()
