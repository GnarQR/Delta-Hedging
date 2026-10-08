# 하나의 장애 상황을 만들고, 진단 Agent가 조사할 수 있게 해주는 환경을 생성
import numpy as np

from simulator import generate_gbm_paths, realized_volatility
from hedge_trace import run_delta_hedge_with_trace

# P&L 중에서 특히 심하게 손실이 난 경우가 어느 정도인지를 측정
# alpha = 0.95면 가장 나쁜 5%의 P&L 평균을 계산함
def _cvar_left(pnl, alpha=0.95):
    cutoff = np.quantile(pnl, 1.0 - alpha)
    tail = pnl[pnl <= cutoff]
    return float(np.mean(tail))

# 하나의 hedging 시뮬레이션 결과를 요약
def _summary(result):
    pnl = result["pnl"]
    return {
        "mean_pnl": float(np.mean(pnl)),
        "rmse_pnl": float(np.sqrt(np.mean(pnl**2))),
        "cvar95_pnl": _cvar_left(pnl),
        "mean_transaction_cost": float(np.mean(result["transaction_cost"])),
    }

# 하나의 객체가 '하나의 장애 사건(incident)'를 나타냄
class HedgingIncidentEnv:
    def __init__(  # Incident 생성 (Class를 만들 때 자동으로 실행되는 초기화 함수)
        self,
        incident_type="vol_misspec",
        seed=11,
        n_paths=3000,
        fault_params=None,  # fault 강도를 외부에서 덮어쓰기 위한 선택적 파라미터
        market_params=None,  # 시장 regime을 외부에서 덮어쓰기 위한 선택적 파라미터
        audit_path_index=0,  # raw audit trail을 기록할 대표 path
    ):
        self.seed = seed
        self.incident_type = incident_type
        self.fault_params = {} if fault_params is None else dict(fault_params)
        self.market_params = {} if market_params is None else dict(market_params)
        self.audit_path_index = int(audit_path_index)

        self.market = {  # 시장의 실제 조건
            "S0": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.03,
            "q": 0.0,
            "sigma_true": 0.20,
            "n_steps": 252,
            "n_paths": n_paths,
            "transaction_cost_rate": 0.0005,
        }

        # market regime에서 바꿀 수 있는 값
        allowed_market_params = {
            "S0", "K", "T", "r", "q",
            "sigma_true", "n_steps",
            "transaction_cost_rate",
        }

        unknown_market = set(self.market_params) - allowed_market_params
        if unknown_market:
            raise ValueError(
                f"Unknown market_params: {sorted(unknown_market)}"
            )

        # 외부 market regime 적용
        self.market.update(self.market_params)
        self.market["n_paths"] = n_paths
        self._validate_market()

        # 중요: sigma_true가 바뀌면 정상 hedger의 sigma_hedge도 같이 바뀌어야 함
        self.healthy = {  # 정상 설정
            "sigma_hedge": 0.20,
            "rebalance_every": 5,
            "observation_lag_steps": 0,
            "execution_lag_steps": 0,
        }

        # incident마다 외부에서 바꿀 수 있도록 허용할 파라미터
        allowed_fault_params = {
            "normal": set(),
            "vol_misspec": {"sigma_hedge", "sigma_hedge_ratio"},
            "under_rebalance": {"rebalance_every"},
            "stale_observation": {"observation_lag_steps"},
            "execution_delay": {"execution_lag_steps"},
        }

        if incident_type not in allowed_fault_params:
            raise ValueError(f"Unknown incident type: {incident_type}")

        unknown_params = (
            set(self.fault_params.keys())
            - allowed_fault_params[incident_type]
        )

        if unknown_params:
            raise ValueError(
                f"{incident_type} does not accept fault_params "
                f"{sorted(unknown_params)}. "
                f"Allowed: {sorted(allowed_fault_params[incident_type])}"
            )

        # ------------------------------------------------------------
        # 기본 incident 설정
        # fault_params를 주지 않으면 기존 pilot과 완전히 같은 값 사용
        # ------------------------------------------------------------
        if incident_type == "vol_misspec":
            self.hidden_root_cause = {
                "type": "volatility_misspecification",
                "detail": "hedger volatility is too low",
            }
            self.active = dict(self.healthy)

            # 기존 base case(0.20 -> 0.12)는 true sigma의 60%를 사용한 것과 같음.
            # market regime이 바뀌어도 상대적 오류 강도를 유지하도록 ratio 기반으로 처리.
            self.active["sigma_hedge"] = (
                self.market["sigma_true"] * 0.60
            )

            if "sigma_hedge_ratio" in self.fault_params:
                self.active["sigma_hedge"] = (
                    self.market["sigma_true"]
                    * float(self.fault_params["sigma_hedge_ratio"])
                )
            elif "sigma_hedge" in self.fault_params:
                self.active["sigma_hedge"] = float(
                    self.fault_params["sigma_hedge"]
                )

        elif incident_type == "under_rebalance":
            self.hidden_root_cause = {
                "type": "under_rebalancing",
                "detail": "hedge is updated too infrequently",
            }
            self.active = dict(self.healthy)
            self.active["rebalance_every"] = 20

        elif incident_type == "stale_observation":
            self.hidden_root_cause = {
                "type": "stale_market_observation",
                "detail": "Delta is calculated from delayed stock prices",
            }
            self.active = dict(self.healthy)
            self.active["observation_lag_steps"] = 5

        elif incident_type == "execution_delay":
            self.hidden_root_cause = {
                "type": "execution_delay",
                "detail":
                    "Delta is calculated on time, but hedge orders are executed later",
            }
            self.active = dict(self.healthy)
            self.active["execution_lag_steps"] = 5

        elif incident_type == "normal":
            self.hidden_root_cause = {
                "type": "no_fault",
                "detail": "healthy baseline configuration",
            }
            self.active = dict(self.healthy)

        # ------------------------------------------------------------
        # 외부에서 전달된 fault severity로 기본값을 덮어씀
        #
        # 예:
        # HedgingIncidentEnv(
        #     incident_type="stale_observation",
        #     fault_params={"observation_lag_steps": 3},
        # )
        # ------------------------------------------------------------
        # vol_misspec은 위에서 sigma_hedge / sigma_hedge_ratio를 별도 처리함
        if incident_type != "vol_misspec":
            self.active.update(self.fault_params)

        self._validate_active_config()

        # 시장을 한 번 생성 후 저장
        self.paths = generate_gbm_paths(
            S0=self.market["S0"],
            r=self.market["r"],
            sigma=self.market["sigma_true"],
            T=self.market["T"],
            n_steps=self.market["n_steps"],
            n_paths=self.market["n_paths"],
            seed=self.seed,
            q=self.market["q"],
        )

        # 장애 상태에서 hedging 실행
        self.result = self._run(self.active)


    def _validate_market(self):
        if self.market["S0"] <= 0:
            raise ValueError("S0 must be > 0")
        if self.market["K"] <= 0:
            raise ValueError("K must be > 0")
        if self.market["T"] <= 0:
            raise ValueError("T must be > 0")
        if self.market["sigma_true"] <= 0:
            raise ValueError("sigma_true must be > 0")
        if int(self.market["n_steps"]) < 2:
            raise ValueError("n_steps must be >= 2")
        if self.market["transaction_cost_rate"] < 0:
            raise ValueError("transaction_cost_rate must be >= 0")
        self.market["n_steps"] = int(self.market["n_steps"])


    # Severity sweep에서 잘못된 값을 넣었을 때 이상한 결과가 나오지 않도록 방지
    def _validate_active_config(self):
        if self.active["sigma_hedge"] <= 0:
            raise ValueError("sigma_hedge must be > 0")

        if int(self.active["rebalance_every"]) < 1:
            raise ValueError("rebalance_every must be >= 1")

        if int(self.active["observation_lag_steps"]) < 0:
            raise ValueError("observation_lag_steps must be >= 0")

        if int(self.active["execution_lag_steps"]) < 0:
            raise ValueError("execution_lag_steps must be >= 0")

        # step 관련 파라미터는 정수로 통일
        self.active["rebalance_every"] = int(
            self.active["rebalance_every"]
        )
        self.active["observation_lag_steps"] = int(
            self.active["observation_lag_steps"]
        )
        self.active["execution_lag_steps"] = int(
            self.active["execution_lag_steps"]
        )


    # 반복해서 사용할 수 있도록 hedging 실행 부분을 따로 작성
    def _run(self, config):
        return run_delta_hedge_with_trace(
            paths=self.paths,  # 항상 똑같은 시장 경로를 사용
            K=self.market["K"],
            T=self.market["T"],
            r=self.market["r"],
            sigma_market=self.market["sigma_true"],
            sigma_hedge=config["sigma_hedge"],
            q=self.market["q"],
            rebalance_every=config["rebalance_every"],
            transaction_cost_rate=self.market["transaction_cost_rate"],
            observation_lag_steps=config["observation_lag_steps"],
            execution_lag_steps=config["execution_lag_steps"],
            audit_path_index=self.audit_path_index,
        )


    # Agent가 사용할 수 있는 진단 도구 목록    
    '''  # 기존 함수
    def available_tools(self):
        return {
            "inspect_market":
                "Inspect realized market behavior.",
            "inspect_model_config":
                "Inspect numerical settings used by the hedger.",
            "inspect_trade_activity":
                "Inspect hedge trading activity.",
            "inspect_delta_trajectory":
                "Inspect Delta mismatch through time.",
            "inspect_observation_alignment":
                "Inspect whether observed prices track current prices.",
            "inspect_pnl_trajectory":
                "Inspect coarse hedge portfolio trajectory.",
            "run_counterfactual":
                "Change one mechanism and re-run the SAME paths.",
        }
    '''
    # 원인을 요약해서 알려주는 기존 inspect_* 도구 대신,
    # 실제 audit trail에 가까운 raw records를 제공.
    def available_tools(self):
        return {
            "get_contract_context":
                "Inspect contract and hedge environment context.",
            "get_market_history":
                "Read recorded stock-price history for a requested time window.",
            "get_model_decision_history":
                "Read model input/output records at hedge-decision times.",
            "get_order_history":
                "Read hedge order submission records.",
            "get_fill_history":
                "Read actual trade fill records.",
            "get_position_history":
                "Read hedge position and cash-account history.",
            "estimate_realized_volatility":
                "Estimate realized volatility from recorded market prices.",
            "run_counterfactual":
                "Change one mechanism and re-run the SAME market paths.",
        }

    # Agent가 처음에 받는 정보
    def initial_observation(self):
        s = _summary(self.result)  # 현재 장애 결과 요약
        return {  # LLM agent에게 보여주는 초기 증상
            "incident": "Hedging diagnostic episode",
            "mean_pnl": round(s["mean_pnl"], 4),
            "rmse_pnl": round(s["rmse_pnl"], 4),
            "cvar95_pnl": round(s["cvar95_pnl"], 4),
        }


    # -----------------------------------------------------------------
    # Agent-facing raw audit trail tools
    # -----------------------------------------------------------------
    # 실제 계약 / hedge 설정에서 알 수 있는 정적 정보.
    def get_contract_context(self):
        return {
            "initial_stock_price":
                round(
                    float(
                        self.market["S0"]
                    ),
                    4,
                ),
            "strike":
                round(
                    float(
                        self.market["K"]
                    ),
                    4,
                ),
            "maturity_years":
                round(
                    float(
                        self.market["T"]
                    ),
                    6,
                ),
            "risk_free_rate":
                round(
                    float(
                        self.market["r"]
                    ),
                    6,
                ),
            "dividend_yield":
                round(
                    float(
                        self.market["q"]
                    ),
                    6,
                ),
            "transaction_cost_rate":
                round(
                    float(
                        self.market[
                            "transaction_cost_rate"
                        ]
                    ),
                    8,
                ),
            "n_steps":
                int(
                    self.market["n_steps"]
                ),
        }


    # Audit record 공통 slicing helper, 너무 많은 record를 한 번에 agent에게 주지 않도록 최대 100개로 제한
    def _filter_audit_records(
        self,
        records,
        step_field,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        if end_step is None:
            end_step = (
                self.market["n_steps"]
            )

        start_step = int(start_step)
        end_step = int(end_step)
        limit = int(limit)

        if start_step < 0:
            raise ValueError(
                "start_step must be >= 0"
            )

        if end_step < start_step:
            raise ValueError(
                "end_step must be >= start_step"
            )

        if limit < 1:
            raise ValueError(
                "limit must be >= 1"
            )

        limit = min(limit, 100,)

        selected = [
            row
            for row in records
            if (
                start_step
                <= int(
                    row[step_field]
                )
                <= end_step
            )
        ]

        truncated = (
            len(selected)
            > limit
        )

        return {
            "start_step":
                start_step,
            "end_step":
                end_step,
            "returned_count":
                min(
                    len(selected),
                    limit,
                ),
            "truncated":
                truncated,
            "records":
                selected[:limit],
        }


    # 시장가격 기록
    def get_market_history(
        self,
        start_step=0,
        end_step=None,
        stride=1,
        limit=40,
    ):
        records = (
            self.result[
                "audit_log"
            ][
                "market_history"
            ]
        )

        if end_step is None:
            end_step = (
                self.market[
                    "n_steps"
                ]
            )

        stride = int(stride)

        if stride < 1:
            raise ValueError(
                "stride must be >= 1"
            )

        filtered = [
            row
            for row in records
            if (
                int(start_step)
                <= int(
                    row["step"]
                )
                <= int(end_step)
                and (
                    int(
                        row["step"]
                    )
                    - int(start_step)
                )
                % stride
                == 0
            )
        ]

        return self._filter_audit_records(
            filtered,
            step_field="step",
            start_step=start_step,
            end_step=end_step,
            limit=limit,
        )


    # Delta 계산 시점의 raw model log (관측된 주식 가격은 보여주지만 몇 step 전 가격인지 정답은 알려주지 않음)
    def get_model_decision_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self._filter_audit_records(
            self.result[
                "audit_log"
            ][
                "model_decisions"
            ],
            step_field=
                "decision_step",
            start_step=
                start_step,
            end_step=
                end_step,
            limit=
                limit,
        )


    # 주문 제출 기록
    def get_order_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self._filter_audit_records(
            self.result[
                "audit_log"
            ][
                "orders"
            ],
            step_field=
                "submitted_step",
            start_step=
                start_step,
            end_step=
                end_step,
            limit=
                limit,
        )


    # 실제 체결 기록
    def get_fill_history(
        self,
        start_step=0,
        end_step=None,
        limit=40,
    ):
        return self._filter_audit_records(
            self.result[
                "audit_log"
            ][
                "fills"
            ],
            step_field=
                "fill_step",
            start_step=
                start_step,
            end_step=
                end_step,
            limit=
                limit,
        )


    # 실제 hedge position / cash / hedge portfolio value 기록
    def get_position_history(
        self,
        start_step=0,
        end_step=None,
        stride=1,
        limit=40,
    ):
        records = (
            self.result[
                "audit_log"
            ][
                "position_history"
            ]
        )

        if end_step is None:
            end_step = (
                self.market[
                    "n_steps"
                ]
            )

        stride = int(stride)

        if stride < 1:
            raise ValueError(
                "stride must be >= 1"
            )

        filtered = [
            row
            for row in records
            if (
                int(start_step)
                <= int(
                    row["step"]
                )
                <= int(end_step)
                and (
                    int(
                        row["step"]
                    )
                    - int(start_step)
                )
                % stride
                == 0
            )
        ]

        return self._filter_audit_records(
            filtered,
            step_field="step",
            start_step=start_step,
            end_step=end_step,
            limit=limit,
        )


    # 실제 실현 변동성 추정(sigma_true를 알려주지 않고, audit path의 실제 가격 기록에서 계산)
    def estimate_realized_volatility(
        self,
        start_step=0,
        end_step=None,
    ):
        if end_step is None:
            end_step = (
                self.market[
                    "n_steps"
                ]
            )

        start_step = int(start_step)
        end_step = int(end_step)

        if (
            start_step < 0
            or end_step
            > self.market[
                "n_steps"
            ]
            or end_step
            <= start_step
        ):
            raise ValueError(
                "Invalid volatility window"
            )

        prices = self.paths[
            self.audit_path_index,
            start_step:
                end_step + 1,
        ]

        log_returns = np.diff(
            np.log(prices)
        )

        if len(log_returns) < 2:
            raise ValueError(
                "Need at least 2 returns"
            )

        dt = (
            self.market["T"]
            / self.market[
                "n_steps"
            ]
        )

        annualized = (
            np.std(
                log_returns,
                ddof=1,
            )
            / np.sqrt(dt)
        )

        return {
            "start_step":
                start_step,
            "end_step":
                end_step,
            "n_returns":
                int(
                    len(log_returns)
                ),
            "annualized_realized_volatility":
                round(
                    float(
                        annualized
                    ),
                    6,
                ),
        }


    # Agent가 시장을 조사할 때 사용하는 tool
    def inspect_market(self):
        return {
            "S0": float(self.market["S0"]),
            "K": float(self.market["K"]),
            "moneyness_S_over_K": round(
                self.market["S0"] / self.market["K"], 4
            ),
            "T": float(self.market["T"]),
            "realized_volatility_estimate": round(
                realized_volatility(self.paths, self.market["T"]), 4
            ),
            "mean_terminal_stock": round(
                float(np.mean(self.paths[:, -1])), 4
            ),
            "stock_p05": round(
                float(np.quantile(self.paths[:, -1], 0.05)), 4
            ),
            "stock_p95": round(
                float(np.quantile(self.paths[:, -1], 0.95)), 4
            ),
        }


    # 현재 hedger가 사용하는 설정을 보여줌
    def inspect_model_config(self):
        return {  # 현재 설정값을 보여줄 뿐, agent가 시장 evidence와 비교해서 진단해야 함
            "hedger_volatility_input": self.active["sigma_hedge"],
            "rebalance_every_steps": self.active["rebalance_every"],
            "transaction_cost_rate": self.market["transaction_cost_rate"],
        }


    def inspect_trade_activity(self):
        trace = self.result["trace"]

        rebalance_steps = [
            i for i, flag in enumerate(trace["rebalance_flag"])
            if flag == 1
        ]

        return {
            "mean_turnover_dollars": round(
                float(np.mean(self.result["turnover"])), 4
            ),
            "mean_transaction_cost": round(
                float(np.mean(self.result["transaction_cost"])), 4
            ),
            "rebalance_event_count": len(rebalance_steps),
            "first_rebalance_steps": rebalance_steps[:10],
            "peak_mean_trade_notional": round(
                float(np.max(trace["mean_trade_notional"])), 4
            ),
        }


    def inspect_delta_trajectory(self):
        gap = np.asarray(self.result["trace"]["mean_abs_delta_gap"])
        n = len(gap)

        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        windows = np.array_split(gap, 8)

        return {
            "mean_abs_delta_gap": round(float(np.mean(gap)), 6),
            "max_abs_delta_gap": round(float(np.max(gap)), 6),
            "delta_gap_checkpoints": [
                {
                    "step": int(i),
                    "mean_abs_delta_gap": round(float(gap[i]), 6),
                }
                for i in checkpoints
            ],
            "delta_gap_by_window": [
                round(float(np.mean(w)), 6) for w in windows
            ],
        }


    def inspect_observation_alignment(self):
        gap = np.asarray(
            self.result["trace"]["mean_abs_price_observation_gap"]
        )
        windows = np.array_split(gap, 8)

        return {
            "mean_abs_price_observation_gap":
                round(float(np.mean(gap)), 4),
            "max_abs_price_observation_gap":
                round(float(np.max(gap)), 4),
            "observation_gap_by_window": [
                round(float(np.mean(w)), 4) for w in windows
            ],
        }


    # Hedge 주문을 결정한 시점과 실제 포지션이 변경된 시점을 비교하는 tool
    def inspect_execution_timing(self):
        trace = self.result["trace"]

        decision_steps = [
            i for i, flag in enumerate(trace["rebalance_decision_flag"])
            if flag == 1
        ]
        execution_steps = [
            i for i, flag in enumerate(trace["execution_flag"])
            if flag == 1
        ]

        return {
            "decision_event_count": len(decision_steps),
            "execution_event_count": len(execution_steps),
            "first_decision_steps": decision_steps[:10],
            "first_execution_steps": execution_steps[:10],
            "max_pending_orders": int(max(trace["pending_order_count"])),
        }


    # P&L / 포트폴리오가 시간에 따라 어떻게 움직였는지 보여주는 tool
    def inspect_pnl_trajectory(self):
        trace = self.result["trace"]
        pv = np.asarray(trace["mean_portfolio_value"])  # 전체 경로의 평균 hedge portfolio value
        n = len(pv)

        points = [0, n // 4, n // 2, 3 * n // 4, n - 1]  # 다섯 지점만 선택 (요약된 checkpoint만 제공 -> 정보량 제한)
        
        return {
            "portfolio_value_checkpoints": [
                {
                    "step": int(i),
                    "mean_hedge_portfolio_value": round(float(pv[i]), 4),
                }
                for i in points
            ],
            "final_summary": {
                k: round(v, 4) for k, v in _summary(self.result).items()
            },
        }


    # Agent가 counterfactual 실험을 할 때 사용하는 tool
    def run_counterfactual(self, intervention):
        cfg = dict(self.active)  # 원본 incident를 건드리지 않고 현재 설정을 복사

        if intervention == "use_realized_volatility":
            cfg["sigma_hedge"] = realized_volatility(
                self.paths, self.market["T"]
            )

        elif intervention == "use_reference_volatility":
            cfg["sigma_hedge"] = self.market["sigma_true"]

        elif intervention == "rebalance_more_frequently":
            cfg["rebalance_every"] = max(1, self.active["rebalance_every"] // 4)

        elif intervention == "rebalance_less_frequently":
            cfg["rebalance_every"] = min(
                self.market["n_steps"],
                self.active["rebalance_every"] * 4,
            )

        elif intervention == "remove_observation_delay":
            cfg["observation_lag_steps"] = 0

        elif intervention == "remove_execution_delay":
            cfg["execution_lag_steps"] = 0

        else:
            raise ValueError(f"Unknown intervention: {intervention}")

        cf = self._run(cfg)

        before = _summary(self.result)
        after = _summary(cf)  # Before vs After 비교

        return {
            "intervention": intervention,
            "tested_config": {
                "sigma_hedge": round(float(cfg["sigma_hedge"]), 4),
                "rebalance_every_steps": int(cfg["rebalance_every"]),
                "observation_lag_steps": int(cfg["observation_lag_steps"]),
                "execution_lag_steps": int(cfg["execution_lag_steps"]),
            },
            "before": {k: round(v, 4) for k, v in before.items()},
            "after": {k: round(v, 4) for k, v in after.items()},
            "rmse_change": round(
                after["rmse_pnl"] - before["rmse_pnl"], 4
            ),
            "cvar95_change": round(
                after["cvar95_pnl"] - before["cvar95_pnl"], 4
            ),
        }

    # Agent가 절대 보면 안 되는 정답지
    def evaluator_ground_truth(self):
        return self.hidden_root_cause
