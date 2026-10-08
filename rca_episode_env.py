import numpy as np

from bs import bs_call_price
from incident_env import HedgingIncidentEnv


def _scalar(x):
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


class RCAEpisodeEnv:
    """
    One coherent RCA episode:
    one market trajectory = one diagnostic incident.

    diagnosis_step=None:
        post-mortem diagnosis at maturity.

    diagnosis_step=<int>:
        only information up to that step is visible.
    """

    def __init__(
        self,
        incident_type="execution_delay",
        seed=1001,
        fault_params=None,
        market_params=None,
        diagnosis_step=None,
        episode_id=None,
    ):
        self.incident_type = incident_type
        self.seed = int(seed)
        self.fault_params = {} if fault_params is None else dict(fault_params)
        self.market_params = {} if market_params is None else dict(market_params)

        # Reuse the already-validated fault / market machinery.
        # n_paths=1 makes this a single realized market trajectory.
        self.base = HedgingIncidentEnv(
            incident_type=incident_type,
            seed=self.seed,
            n_paths=1,
            fault_params=self.fault_params,
            market_params=self.market_params,
            audit_path_index=0,
        )

        self.market = self.base.market
        self.active = self.base.active
        self.result = self.base.result

        n_steps = int(self.market["n_steps"])

        if diagnosis_step is None:
            diagnosis_step = n_steps

        self.diagnosis_step = int(diagnosis_step)

        if not 1 <= self.diagnosis_step <= n_steps:
            raise ValueError(
                f"diagnosis_step must be between 1 and {n_steps}"
            )

        self.episode_id = (
            str(episode_id)
            if episode_id is not None
            else f"{incident_type}_seed{self.seed}_t{self.diagnosis_step}"
        )

        self._pnl_history = self._build_marked_pnl_history(self.result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_end(self, end_step):
        if end_step is None:
            return self.diagnosis_step
        return min(int(end_step), self.diagnosis_step)

    def _build_marked_pnl_history(self, result):
        market_records = result["audit_log"]["market_history"]
        position_records = result["audit_log"]["position_history"]
        fill_records = result["audit_log"]["fills"]

        market_by_step = {int(x["step"]): x for x in market_records}
        position_by_step = {int(x["step"]): x for x in position_records}

        cost_by_step = {}
        for row in fill_records:
            step = int(row["fill_step"])
            cost_by_step[step] = (
                cost_by_step.get(step, 0.0)
                + float(row["transaction_cost"])
            )

        dt = self.market["T"] / self.market["n_steps"]
        cumulative_cost = 0.0
        history = []

        for step in range(self.market["n_steps"] + 1):
            if step not in market_by_step or step not in position_by_step:
                continue

            S = float(market_by_step[step]["stock_price"])
            position = float(position_by_step[step]["position"])
            cash = float(position_by_step[step]["cash"])

            cumulative_cost += cost_by_step.get(step, 0.0)

            tau = max(self.market["T"] - step * dt, 0.0)

            # Market/reference option mark used only to construct observed P&L.
            # sigma_true itself is NOT shown to the agent.
            option_mark = _scalar(
                bs_call_price(
                    S,
                    self.market["K"],
                    tau,
                    self.market["r"],
                    self.market["sigma_true"],
                    self.market["q"],
                )
            )

            marked_pnl = position * S + cash - option_mark

            history.append(
                {
                    "step": int(step),
                    "marked_hedge_pnl": float(marked_pnl),
                    "cumulative_transaction_cost": float(cumulative_cost),
                }
            )

        return history

    def _summary_to_date(self, result=None):
        history = (
            self._pnl_history
            if result is None
            else self._build_marked_pnl_history(result)
        )

        history = [
            row for row in history
            if int(row["step"]) <= self.diagnosis_step
        ]

        pnl = np.asarray(
            [row["marked_hedge_pnl"] for row in history],
            dtype=float,
        )

        lookback = min(20, len(pnl) - 1)
        recent_change = (
            float(pnl[-1] - pnl[-1 - lookback])
            if lookback > 0
            else 0.0
        )

        return {
            "marked_hedge_pnl": float(pnl[-1]),
            "recent_20_step_pnl_change": recent_change,
            "worst_marked_hedge_pnl_to_date": float(np.min(pnl)),
            "best_marked_hedge_pnl_to_date": float(np.max(pnl)),
            "cumulative_transaction_cost": float(
                history[-1]["cumulative_transaction_cost"]
            ),
        }

    # ------------------------------------------------------------------
    # Agent-facing interface
    # ------------------------------------------------------------------

    def initial_observation(self):
        s = self._summary_to_date()

        return {
            "episode_id": self.episode_id,
            "diagnosis_step": self.diagnosis_step,
            "marked_hedge_pnl": round(s["marked_hedge_pnl"], 6),
            "recent_20_step_pnl_change": round(
                s["recent_20_step_pnl_change"], 6
            ),
            "worst_marked_hedge_pnl_to_date": round(
                s["worst_marked_hedge_pnl_to_date"], 6
            ),
            "cumulative_transaction_cost": round(
                s["cumulative_transaction_cost"], 6
            ),
        }

    def available_tools(self):
        return {
            "get_contract_context":
                "Inspect contract and diagnosis timing context.",
            "get_market_history":
                "Read recorded market prices.",
            "get_model_decision_history":
                "Read Delta-model decision records.",
            "get_order_history":
                "Read hedge order submissions.",
            "get_fill_history":
                "Read actual trade fills.",
            "get_position_history":
                "Read hedge position and cash history.",
            "get_pnl_history":
                "Read marked hedge P&L history.",
            "estimate_realized_volatility":
                "Estimate realized volatility from observed prices.",
            "run_counterfactual":
                "Modify one mechanism and replay the SAME market path.",
        }

    def get_contract_context(self):
        out = dict(self.base.get_contract_context())
        out.update(
            {
                "episode_id": self.episode_id,
                "diagnosis_step": self.diagnosis_step,
                "post_mortem": bool(
                    self.diagnosis_step == self.market["n_steps"]
                ),
            }
        )
        return out

    def get_market_history(
        self,
        start_step=0,
        end_step=None,
        stride=1,
        limit=40,
    ):
        return self.base.get_market_history(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
            stride=stride,
            limit=limit,
        )

    def get_model_decision_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self.base.get_model_decision_history(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
            limit=limit,
        )

    def get_order_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self.base.get_order_history(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
            limit=limit,
        )

    def get_fill_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self.base.get_fill_history(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
            limit=limit,
        )

    def get_position_history(
        self,
        start_step=0,
        end_step=None,
        stride=1,
        limit=40,
    ):
        return self.base.get_position_history(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
            stride=stride,
            limit=limit,
        )

    def get_pnl_history(
        self,
        start_step=0,
        end_step=None,
        stride=1,
        limit=40,
    ):
        start_step = int(start_step)
        end_step = self._clamp_end(end_step)
        stride = int(stride)
        limit = min(int(limit), 100)

        if start_step < 0:
            raise ValueError("start_step must be >= 0")
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        selected = [
            row
            for row in self._pnl_history
            if (
                start_step <= int(row["step"]) <= end_step
                and (int(row["step"]) - start_step) % stride == 0
            )
        ]

        return {
            "start_step": start_step,
            "end_step": end_step,
            "returned_count": min(len(selected), limit),
            "truncated": len(selected) > limit,
            "records": selected[:limit],
        }

    def estimate_realized_volatility(
        self,
        start_step=0,
        end_step=None,
    ):
        return self.base.estimate_realized_volatility(
            start_step=start_step,
            end_step=self._clamp_end(end_step),
        )

    # ------------------------------------------------------------------
    # Counterfactual replay
    # ------------------------------------------------------------------

    def run_counterfactual(self, intervention, value=None):
        cfg = dict(self.active)

        if intervention == "use_realized_volatility":
            est = self.estimate_realized_volatility(
                0,
                self.diagnosis_step,
            )["annualized_realized_volatility"]
            cfg["sigma_hedge"] = float(est)

        elif intervention == "set_volatility_input":
            if value is None:
                raise ValueError(
                    "value is required for set_volatility_input"
                )
            value = float(value)
            if value <= 0:
                raise ValueError("volatility must be > 0")
            cfg["sigma_hedge"] = value

        elif intervention == "set_rebalance_every":
            if value is None:
                raise ValueError(
                    "value is required for set_rebalance_every"
                )
            value = int(value)
            if value < 1:
                raise ValueError("rebalance interval must be >= 1")
            cfg["rebalance_every"] = value

        elif intervention == "remove_observation_delay":
            cfg["observation_lag_steps"] = 0

        elif intervention == "remove_execution_delay":
            cfg["execution_lag_steps"] = 0

        else:
            raise ValueError(
                f"Unknown intervention: {intervention}"
            )

        # Same exact market trajectory.
        cf_result = self.base._run(cfg)

        before = self._summary_to_date(self.result)
        after = self._summary_to_date(cf_result)

        before_error = abs(before["marked_hedge_pnl"])
        after_error = abs(after["marked_hedge_pnl"])

        return {
            "intervention": intervention,
            "value": value,
            "diagnosis_step": self.diagnosis_step,
            "before": {
                k: round(float(v), 6)
                for k, v in before.items()
            },
            "after": {
                k: round(float(v), 6)
                for k, v in after.items()
            },
            "single_path_pnl_change": round(
                after["marked_hedge_pnl"]
                - before["marked_hedge_pnl"],
                6,
            ),
            "single_path_absolute_pnl_error_change": round(
                after_error - before_error,
                6,
            ),
            "note": (
                "Single-path P&L is noisy. "
                "A correct repair is not guaranteed to improve "
                "terminal P&L on every individual path."
            ),
        }

    # ------------------------------------------------------------------
    # Evaluator only
    # ------------------------------------------------------------------

    def _reference_tracking_error(self, result):
        """
        Simulator-only oracle metric.

        mean_abs_delta_gap is measured against the healthy/reference
        current-price Delta. This is useful for benchmark scoring,
        but should NOT be exposed to the LLM agent.
        """
        gap = np.asarray(
            result["trace"]["mean_abs_delta_gap"],
            dtype=float,
        )

        # trace has steps 0..n_steps-1.
        end = min(
            self.diagnosis_step,
            len(gap) - 1,
        )

        return float(
            np.mean(
                gap[: end + 1]
            )
        )

    def evaluator_counterfactual_effect(
        self,
        intervention,
        value=None,
    ):
        """
        Evaluator-only causal verification.

        Uses hidden simulator reference Delta, so never expose this
        result to the agent.
        """
        cfg = dict(self.active)

        if intervention == "use_realized_volatility":
            est = self.estimate_realized_volatility(
                0,
                self.diagnosis_step,
            )["annualized_realized_volatility"]
            cfg["sigma_hedge"] = float(est)

        elif intervention == "set_volatility_input":
            if value is None:
                raise ValueError(
                    "value is required for set_volatility_input"
                )
            cfg["sigma_hedge"] = float(value)

        elif intervention == "set_rebalance_every":
            if value is None:
                raise ValueError(
                    "value is required for set_rebalance_every"
                )
            cfg["rebalance_every"] = int(value)

        elif intervention == "remove_observation_delay":
            cfg["observation_lag_steps"] = 0

        elif intervention == "remove_execution_delay":
            cfg["execution_lag_steps"] = 0

        else:
            raise ValueError(
                f"Unknown intervention: {intervention}"
            )

        cf_result = self.base._run(cfg)

        before = self._reference_tracking_error(
            self.result
        )
        after = self._reference_tracking_error(
            cf_result
        )

        return {
            "intervention": intervention,
            "value": value,
            "reference_tracking_error_before": round(
                before, 6
            ),
            "reference_tracking_error_after": round(
                after, 6
            ),
            "reference_tracking_error_reduction": round(
                before - after, 6
            ),
        }

    def evaluator_ground_truth(self):
        return {
            "episode_id": self.episode_id,
            "root_cause": self.base.evaluator_ground_truth(),
            "incident_type": self.incident_type,
            "fault_params": dict(self.fault_params),
            "market_params": dict(self.market_params),
            "seed": self.seed,
            "diagnosis_step": self.diagnosis_step,
        }
