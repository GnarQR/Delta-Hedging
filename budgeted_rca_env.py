import copy
import hashlib


EVIDENCE_TOOLS = {
    "get_market_history",
    "get_model_decision_history",
    "get_order_history",
    "get_fill_history",
    "get_position_history",
    "get_pnl_history",
    "estimate_realized_volatility",
}

RECORD_TOOLS = {
    "get_market_history",
    "get_model_decision_history",
    "get_order_history",
    "get_fill_history",
    "get_position_history",
    "get_pnl_history",
}


class BudgetExceededError(RuntimeError):
    pass


class CounterfactualStageError(RuntimeError):
    pass


class RepeatedEvidenceToolError(RuntimeError):
    pass


class BudgetedRCAEnv:
    """
    RCAEpisodeEnv 위에 evidence-access policy를 강제하는 wrapper.

    v0.2 changes
    ------------
    1. Agent-visible episode_id 제거.
       기존 episode_id에는 root-cause label이 포함되어 있어 label leakage가 가능했음.

    2. 같은 evidence family를 반복 pagination하는 것을 기본적으로 금지.
       max_queries_per_tool=1.

    3. raw tool의 `truncated` metadata를 agent에게 보여주지 않음.
       record cap은 인터페이스 제약이지 fault evidence가 아니기 때문.

    4. expected hedge timing policy를 free context로 모든 protocol에 동일하게 제공.

    5. evaluator-only metadata와 agent-visible payload를 명확히 분리.
    """

    def __init__(
        self,
        env,
        evidence_budget=4,
        max_records_per_query=12,
        counterfactual_budget=1,
        max_queries_per_tool=1,
    ):
        self.env = env

        self.evidence_budget = int(evidence_budget)
        self.max_records_per_query = int(max_records_per_query)
        self.counterfactual_budget = int(counterfactual_budget)
        self.max_queries_per_tool = int(max_queries_per_tool)

        if self.evidence_budget < 0:
            raise ValueError("evidence_budget must be >= 0")

        if self.max_records_per_query < 1:
            raise ValueError("max_records_per_query must be >= 1")

        if self.counterfactual_budget < 0:
            raise ValueError("counterfactual_budget must be >= 0")

        if self.max_queries_per_tool < 1:
            raise ValueError("max_queries_per_tool must be >= 1")

        self.evidence_used = 0
        self.counterfactual_used = 0

        self.query_log = []
        self.counterfactual_log = []

        self.provisional_diagnosis = None
        self.counterfactual_stage_open = False

        self.tool_use_counts = {
            name: 0
            for name in EVIDENCE_TOOLS
        }

    # ------------------------------------------------------------
    # Free context
    # ------------------------------------------------------------

    def _public_case_id(self):
        """
        Opaque case identifier.

        Internal env.episode_id contains strings such as:
            execution_delay_seed...
        so it must never be shown to the model.
        """
        raw = str(
            self.env.episode_id
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            raw
        ).hexdigest()[:12]

        return (
            f"case_{digest}"
        )

    def _expected_policy(self):
        """
        Intended operational timing policy, NOT the hidden active config.

        This is necessary for identifiability:
        seeing a hedge decision every 20 steps is not a fault unless the
        analyst knows that the intended interval is 5 steps.
        """
        healthy = (
            self.env.base.healthy
        )

        return {
            "rebalance_every_steps":
                int(
                    healthy[
                        "rebalance_every"
                    ]
                ),
            "observation_delay_steps":
                int(
                    healthy[
                        "observation_lag_steps"
                    ]
                ),
            "execution_delay_steps":
                int(
                    healthy[
                        "execution_lag_steps"
                    ]
                ),
        }

    def initial_packet(self):
        initial_observation = copy.deepcopy(
            self.env.initial_observation()
        )

        contract_context = copy.deepcopy(
            self.env.get_contract_context()
        )

        # Critical v0.2 leakage fix.
        initial_observation.pop(
            "episode_id",
            None,
        )
        contract_context.pop(
            "episode_id",
            None,
        )

        contract_context[
            "expected_hedge_policy"
        ] = self._expected_policy()

        return {
            "case_id":
                self._public_case_id(),
            "initial_observation":
                initial_observation,
            "contract_context":
                contract_context,
            "interface_semantics": {
                "record_cap":
                    (
                        "Each record-returning query is capped by the interface. "
                        "A capped response is NOT evidence of missing, delayed, "
                        "corrupted, or stale source data."
                    ),
                "repeat_policy":
                    (
                        "By default each evidence tool may be used once. "
                        "Choose a targeted window rather than paginating one log."
                    ),
                "counterfactual":
                    (
                        "Single-path counterfactual P&L is noisy. "
                        "No improvement on one realized path does NOT by itself "
                        "disprove a causal hypothesis."
                    ),
            },
            "budget": {
                "evidence_queries_total":
                    self.evidence_budget,
                "max_records_per_query":
                    self.max_records_per_query,
                "max_queries_per_tool":
                    self.max_queries_per_tool,
                "counterfactual_queries_total":
                    self.counterfactual_budget,
            },
        }

    def available_evidence_tools(self):
        descriptions = (
            self.env.available_tools()
        )

        return {
            name:
                descriptions[name]
            for name in EVIDENCE_TOOLS
            if name in descriptions
        }

    # ------------------------------------------------------------
    # Budget state
    # ------------------------------------------------------------

    @property
    def evidence_remaining(self):
        return (
            self.evidence_budget
            - self.evidence_used
        )

    @property
    def counterfactual_remaining(self):
        return (
            self.counterfactual_budget
            - self.counterfactual_used
        )

    def budget_status(self):
        return {
            "evidence_used":
                self.evidence_used,
            "evidence_remaining":
                self.evidence_remaining,
            "counterfactual_used":
                self.counterfactual_used,
            "counterfactual_remaining":
                self.counterfactual_remaining,
            "counterfactual_stage_open":
                self.counterfactual_stage_open,
            "tool_use_counts":
                copy.deepcopy(
                    self.tool_use_counts
                ),
        }

    # ------------------------------------------------------------
    # Evidence queries
    # ------------------------------------------------------------

    def _sanitize_record_args(
        self,
        tool_name,
        args,
    ):
        clean = dict(args)

        if tool_name in RECORD_TOOLS:
            requested_limit = int(
                clean.get(
                    "limit",
                    self.max_records_per_query,
                )
            )

            clean[
                "limit"
            ] = min(
                requested_limit,
                self.max_records_per_query,
            )

        return clean

    def _visible_result(
        self,
        raw_result,
    ):
        """
        Remove interface-internal metadata from what the model sees.

        In particular, `truncated=True` previously caused a model to infer
        a stale/missing market feed. That flag only means the UI record cap
        was reached, so it is evaluator metadata, not causal evidence.
        """
        visible = copy.deepcopy(
            raw_result
        )

        visible.pop(
            "truncated",
            None,
        )

        return visible

    def query(
        self,
        tool_name,
        **kwargs,
    ):
        if self.counterfactual_stage_open:
            raise CounterfactualStageError(
                "Evidence stage is closed after provisional diagnosis."
            )

        if tool_name not in EVIDENCE_TOOLS:
            raise ValueError(
                f"Tool is not allowed as evidence: {tool_name}"
            )

        if self.evidence_remaining <= 0:
            raise BudgetExceededError(
                "No evidence queries remaining."
            )

        if (
            self.tool_use_counts[
                tool_name
            ]
            >= self.max_queries_per_tool
        ):
            raise RepeatedEvidenceToolError(
                f"{tool_name} already reached the per-tool query limit "
                f"({self.max_queries_per_tool}). Select a different evidence family."
            )

        if not hasattr(
            self.env,
            tool_name,
        ):
            raise ValueError(
                f"Environment has no tool named {tool_name}"
            )

        clean_args = (
            self._sanitize_record_args(
                tool_name,
                kwargs,
            )
        )

        fn = getattr(
            self.env,
            tool_name,
        )

        raw_result = fn(
            **clean_args
        )

        visible_result = (
            self._visible_result(
                raw_result
            )
        )

        self.evidence_used += 1
        self.tool_use_counts[
            tool_name
        ] += 1

        query_id = (
            f"Q{self.evidence_used}"
        )

        # What the agent gets.
        visible_record = {
            "query_id":
                query_id,
            "tool":
                tool_name,
            "args":
                copy.deepcopy(
                    clean_args
                ),
            "result":
                copy.deepcopy(
                    visible_result
                ),
            "evidence_remaining":
                self.evidence_remaining,
        }

        # What the evaluator stores.
        audit_record = {
            **copy.deepcopy(
                visible_record
            ),
            "interface_metadata": {
                "truncated_by_record_cap":
                    bool(
                        raw_result.get(
                            "truncated",
                            False,
                        )
                    )
                    if isinstance(
                        raw_result,
                        dict,
                    )
                    else False,
            },
        }

        self.query_log.append(
            audit_record
        )

        return copy.deepcopy(
            visible_record
        )

    # ------------------------------------------------------------
    # Provisional diagnosis / counterfactual
    # ------------------------------------------------------------

    def open_counterfactual_stage(
        self,
        provisional_diagnosis,
    ):
        if self.counterfactual_stage_open:
            raise CounterfactualStageError(
                "Counterfactual stage is already open."
            )

        self.provisional_diagnosis = (
            copy.deepcopy(
                provisional_diagnosis
            )
        )
        self.counterfactual_stage_open = True

        return {
            "status":
                "counterfactual_stage_open",
            "counterfactual_remaining":
                self.counterfactual_remaining,
        }

    def run_counterfactual(
        self,
        intervention,
        value=None,
    ):
        if not self.counterfactual_stage_open:
            raise CounterfactualStageError(
                "Submit a provisional diagnosis before counterfactual testing."
            )

        if self.counterfactual_remaining <= 0:
            raise BudgetExceededError(
                "No counterfactual queries remaining."
            )

        raw = (
            self.env.run_counterfactual(
                intervention,
                value,
            )
        )

        result = copy.deepcopy(
            raw
        )

        # Stronger v0.2 interpretation warning.
        result[
            "interpretation_rule"
        ] = (
            "Treat this single-path P&L replay as secondary evidence only. "
            "A correct repair can worsen realized terminal P&L on one path "
            "because hedging P&L is stochastic. Do not accept or reject a "
            "root cause solely from the sign of the P&L change."
        )

        self.counterfactual_used += 1

        record = {
            "counterfactual_id":
                f"CF{self.counterfactual_used}",
            "intervention":
                intervention,
            "value":
                value,
            "result":
                copy.deepcopy(
                    result
                ),
            "counterfactual_remaining":
                self.counterfactual_remaining,
        }

        self.counterfactual_log.append(
            record
        )

        return copy.deepcopy(
            record
        )

    # ------------------------------------------------------------
    # Experiment record
    # ------------------------------------------------------------

    def audit_record(self):
        return {
            # Internal ID is allowed in evaluator output, not agent input.
            "episode_id":
                self.env.episode_id,
            "public_case_id":
                self._public_case_id(),
            "budget":
                self.budget_status(),
            "provisional_diagnosis":
                copy.deepcopy(
                    self.provisional_diagnosis
                ),
            "queries":
                copy.deepcopy(
                    self.query_log
                ),
            "counterfactuals":
                copy.deepcopy(
                    self.counterfactual_log
                ),
        }
