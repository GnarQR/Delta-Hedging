# 옵션 매도 -> Delta 계산 -> 주식 매수 -> 주가 변화 -> Delta 재계산
# -> 주식 추가 매수/매도 -> 거래비용 발생 -> 만기 hedging 손익 계산

import numpy as np
from bs import bs_call_price, bs_call_delta

# 매도한 유럽형 콜옵션 1개를 Delta hedging하는 함수 (콜옵션을 판 사람(short call) 입장)
def run_delta_hedge_with_trace(
    paths,  # GBM 주가 경로들 -> 각 경로에서 실제로 Delta Hedging 수행
    K,
    T,
    r,
    sigma_market,
    sigma_hedge,
    q=0.0,
    rebalance_every=5,
    transaction_cost_rate=0.0005,
    observation_lag_steps=0,  # Delta 계산 시 현재 가격이 아니라 몇 step 전 가격을 사용할지
    execution_lag_steps=0,  # Delta 계산 후 실제 주문이 몇 step 뒤에 체결될지
    audit_path_index=0,  # 실제 RCA용 raw audit trail을 기록할 대표 path
):
    """
    기존 Monte Carlo aggregate 결과 + 한 개 path의 raw audit trail을 함께 기록.

    주의:
    - aggregate trace는 기존 severity/regime 분석용.
    - audit_log는 LLM RCA evidence용.
    - audit_log에는 hidden root cause나 '정답 gap'을 직접 넣지 않음.
    """

    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps

    if rebalance_every < 1:
        raise ValueError("rebalance_every must be >= 1")
    if observation_lag_steps < 0:
        raise ValueError("observation_lag_steps must be >= 0")
    if execution_lag_steps < 0:
        raise ValueError("execution_lag_steps must be >= 0")
    if not (0 <= audit_path_index < n_paths):
        raise ValueError(
            f"audit_path_index must be between 0 and {n_paths - 1}"
        )

    a = int(audit_path_index)

    S0 = paths[:, 0]
    premium = bs_call_price(S0, K, T, r, sigma_market, q)  # 옵션을 매도했을 때 받는 옵션 가격
    delta = bs_call_delta(S0, K, T, r, sigma_hedge, q)  # 콜옵션을 팔았으므로 주식을 Delta만큼 보유해야 함

    # 분석용 aggregate trace에서만 사용하는 reference Delta.
    # Agent용 audit_log에는 직접 노출하지 않음.
    reference_delta = bs_call_delta(S0, K, T, r, sigma_market, q)
    initial_trade_cost = transaction_cost_rate * np.abs(delta) * S0  # 주식을 매수하면서 발생하는 거래비용
    cash = premium - delta * S0 - initial_trade_cost  # 남은 돈은 cash 계좌에 넣음, 마이너스면 부족한 돈을 빌린 것 

    # Rebalancing : Delta가 바뀌어서 주식을 사고파는 행위
    # Turnover : 그 과정에서 얼마나 많이 거래를 했는지 (거래 행동을 돈으로 환산한 비용)
    # Transaction Cost : 그 거래 때문에 얼마의 비용을 지불했는지
    # Delta를 매번 완벽하게 따라간다면 Hedging 오차는 줄지만, 거래가 많아져 turnover와 transaction cost가 커짐
    total_transaction_cost = initial_trade_cost.copy()
    turnover = np.abs(delta) * S0  # 처음 주식을 산 것도 거래, 거래금액(notional) 기준 turnover


    # ------------------------------------------------------------
    # Pending orders
    # execution delay가 있을 경우 아직 체결되지 않은 주문을 저장
    # 각 주문에는 '언제 체결할지'와 '그때 목표로 했던 Delta'를 저장
    # ------------------------------------------------------------
    pending_orders = []
    next_order_id = 1

    # ------------------------------------------------------------
    # 기존 aggregate traces
    # ------------------------------------------------------------
    mean_stock = [float(np.mean(S0))]
    mean_observed_stock = [float(np.mean(S0))]
    mean_abs_price_observation_gap = [0.0]
    mean_abs_position = [float(np.mean(np.abs(delta)))]
    mean_trade_notional = [float(np.mean(np.abs(delta) * S0))]
    mean_cum_cost = [float(np.mean(total_transaction_cost))]
    mean_portfolio_value = [float(np.mean(delta * S0 + cash))]
    mean_abs_delta_gap = [float(np.mean(np.abs(delta - reference_delta)))]
    
    # 기존 rebalance_flag는 '실제로 포지션이 변경(체결)된 시점'을 뜻하도록 유지
    rebalance_flag = [1]

    # 주문 의사결정 시점과 실제 체결 시점을 따로 기록
    rebalance_decision_flag = [1]
    execution_flag = [1]
    pending_order_count = [0]


    # ------------------------------------------------------------
    # Agent용 raw audit trail
    #
    # 서로 다른 source를 분리해서 저장.
    # Agent는 나중에 여러 tool을 호출해서 직접 맞춰봐야 함.
    # ------------------------------------------------------------
    market_history = [
        {
            "step": 0,
            "stock_price": float(S0[a]),
        }
    ]

    model_decisions = [
        {
            "decision_step": 0,
            "observed_stock_price":
                float(S0[a]),
            "target_delta":
                float(delta[a]),
            "volatility_input":
                float(sigma_hedge),
            "time_to_maturity":
                float(T),
        }
    ]

    orders = [
        {
            "order_id": 0,
            "submitted_step": 0,
            "target_position":
                float(delta[a]),
        }
    ]

    fills = [
        {
            "order_id": 0,
            "fill_step": 0,
            "fill_price":
                float(S0[a]),
            "shares_traded":
                float(delta[a]),
            "resulting_position":
                float(delta[a]),
            "transaction_cost":
                float(
                    initial_trade_cost[a]
                ),
        }
    ]

    position_history = [
        {
            "step": 0,
            "position":
                float(delta[a]),
            "cash":
                float(cash[a]),
            "hedge_portfolio_value":
                float(
                    delta[a] * S0[a]
                    + cash[a]
                ),
        }
    ]


    # ------------------------------------------------------------
    # Rebalancing loop
    # ------------------------------------------------------------
    for step in range(1, n_steps):
        cash *= np.exp(r * dt)  # 현금 계좌에 무위험 이자율이 적용(양수면 이자, 음수면 부채)
        
        S_current = paths[:, step]
        tau = T - step * dt

        observed_step = max(0, step - observation_lag_steps)
        S_observed = paths[:, observed_step]

        market_history.append(
            {
                "step": int(step),
                "stock_price":
                    float(S_current[a]),
            }
        )

        traded_notional = np.zeros(n_paths)
        did_execute = 0
        did_decide = 0

        # ------------------------------------------------------------
        # 1) 이전 시점에 계산해 둔 주문 중 오늘 체결되어야 하는 주문을 먼저 실행
        # ------------------------------------------------------------
        due_orders = [order for order in pending_orders if order["execute_step"] == step]
        pending_orders = [order for order in pending_orders if order["execute_step"] != step]

        for order in due_orders:
            target_delta = order["target_delta"]

            # 주문을 냈을 때의 Delta 목표값을 지금 가격에서 실제로 체결
            shares_traded = target_delta - delta
            order_notional = np.abs(shares_traded) * S_current
            trade_cost = transaction_cost_rate * order_notional

            cash -= shares_traded * S_current + trade_cost
            total_transaction_cost += trade_cost
            turnover += order_notional
            traded_notional += order_notional
            delta = target_delta
            did_execute = 1

            fills.append(
                {
                    "order_id":
                        int(
                            order[
                                "order_id"
                            ]
                        ),
                    "fill_step":
                        int(step),
                    "fill_price":
                        float(
                            S_current[a]
                        ),
                    "shares_traded":
                        float(
                            shares_traded[a]
                        ),
                    "resulting_position":
                        float(delta[a]),
                    "transaction_cost":
                        float(
                            trade_cost[a]
                        ),
                }
            )

        # ------------------------------------------------------------
        # 2) 오늘이 Delta 재계산 / 주문 의사결정 시점
        # ------------------------------------------------------------
        if step % rebalance_every == 0:
            did_decide = 1

            # stale observation이 있으면 과거 가격을 사용하고,
            # execution delay만 있는 경우에는 현재 가격을 정상적으로 사용함
            target_delta = bs_call_delta(S_observed, K, tau, r, sigma_hedge, q)

            order_id = next_order_id
            next_order_id += 1

            # 모델 계산 기록.
            # observed_step 자체는 숨김.
            # Agent는 observed_stock_price를 market_history와 직접 비교해야 함.
            model_decisions.append(
                {
                    "decision_step":
                        int(step),
                    "observed_stock_price":
                        float(
                            S_observed[a]
                        ),
                    "target_delta":
                        float(
                            target_delta[a]
                        ),
                    "volatility_input":
                        float(
                            sigma_hedge
                        ),
                    "time_to_maturity":
                        float(tau),
                }
            )

            # 주문 제출 기록.
            # 예정 체결 step은 노출하지 않음.
            orders.append(
                {
                    "order_id":
                        int(order_id),
                    "submitted_step":
                        int(step),
                    "target_position":
                        float(
                            target_delta[a]
                        ),
                }
            )

            if execution_lag_steps == 0:
                # 정상 상황: 계산한 즉시 현재 가격에서 체결
                shares_traded = target_delta - delta
                order_notional = np.abs(shares_traded) * S_current
                trade_cost = transaction_cost_rate * order_notional

                cash -= shares_traded * S_current + trade_cost
                total_transaction_cost += trade_cost
                turnover += order_notional
                traded_notional += order_notional
                delta = target_delta
                did_execute = 1

                fills.append(
                    {
                        "order_id":
                            int(order_id),
                        "fill_step":
                            int(step),
                        "fill_price":
                            float(
                                S_current[a]
                            ),
                        "shares_traded":
                            float(
                                shares_traded[a]
                            ),
                        "resulting_position":
                            float(delta[a]),
                        "transaction_cost":
                            float(
                                trade_cost[a]
                            ),
                    }
                )

            else:
                # Execution delay: Delta 계산은 지금 했지만 실제 주문은 미래에 체결
                execute_step = step + execution_lag_steps

                # 만기 이전에 실제로 체결 가능한 주문만 queue에 저장
                if execute_step < n_steps:
                    pending_orders.append({
                        "order_id": int(order_id),
                        "decision_step": int(step),
                        "execute_step": int(execute_step),
                        "target_delta": target_delta.copy(),
                    })

        reference_delta = bs_call_delta(S_current, K, tau, r, sigma_market, q)

        mean_stock.append(float(np.mean(S_current)))
        mean_observed_stock.append(float(np.mean(S_observed)))
        mean_abs_price_observation_gap.append(float(np.mean(np.abs(S_current - S_observed))))
        mean_abs_position.append(float(np.mean(np.abs(delta))))
        mean_trade_notional.append(float(np.mean(traded_notional)))
        mean_cum_cost.append(float(np.mean(total_transaction_cost)))
        mean_portfolio_value.append(float(np.mean(delta * S_current + cash)))
        mean_abs_delta_gap.append(float(np.mean(np.abs(delta - reference_delta))))

        rebalance_flag.append(did_execute)
        rebalance_decision_flag.append(did_decide)
        execution_flag.append(did_execute)
        pending_order_count.append(len(pending_orders))

        position_history.append(
            {
                "step":
                    int(step),
                "position":
                    float(delta[a]),
                "cash":
                    float(cash[a]),
                "hedge_portfolio_value":
                    float(
                        delta[a]
                        * S_current[a]
                        + cash[a]
                    ),
            }
        )

    # ------------------------------------------------------------
    # Maturity
    # ------------------------------------------------------------
    cash *= np.exp(r * dt)  # 만기 -> 마지막 기간만큼 현금 계좌에 이자를 적용

    ST = paths[:, -1]  # 최종 주가 ST
    payoff = np.maximum(ST - K, 0.0)  # Payoff 계산
    terminal_hedge_value = delta * ST + cash  # 만기 자산 : 주식 delta주 + 현금 cash만큼 보유
    pnl = terminal_hedge_value - payoff  # 손익(Profit & Loss, P&L) 계산
    # Hedging P&L이 0에 가까울수록 Hedging이 잘 된 것이라고 생각

    # maturity market / position snapshot
    market_history.append(
        {
            "step": int(n_steps),
            "stock_price":
                float(ST[a]),
        }
    )

    position_history.append(
        {
            "step": int(n_steps),
            "position":
                float(delta[a]),
            "cash":
                float(cash[a]),
            "hedge_portfolio_value":
                float(
                    terminal_hedge_value[a]
                ),
        }
    )

    return {
        "pnl": pnl,  # hedging 손익/오차
        "transaction_cost": total_transaction_cost,  # 전체 기간 동안 지불한 거래비용
        "turnover": turnover,  # 전체 기간 동안 거래한 주식 금액

        # 기존 분석용 aggregate trace
        "trace": {
            "mean_stock": mean_stock,
            "mean_observed_stock": mean_observed_stock,
            "mean_abs_price_observation_gap": mean_abs_price_observation_gap,
            "mean_abs_position": mean_abs_position,
            "mean_trade_notional": mean_trade_notional,
            "mean_cum_cost": mean_cum_cost,
            "mean_portfolio_value": mean_portfolio_value,
            "mean_abs_delta_gap": mean_abs_delta_gap,
            "rebalance_flag": rebalance_flag,
            "rebalance_decision_flag": rebalance_decision_flag,
            "execution_flag": execution_flag,
            "pending_order_count": pending_order_count,
        },

        # 실제 RCA agent가 조회할 raw audit trail
        "audit_log": {
            "market_history":
                market_history,
            "model_decisions":
                model_decisions,
            "orders":
                orders,
            "fills":
                fills,
            "position_history":
                position_history,
        },

        # evaluator/debug 용도.
        # available_tools()를 통해 agent에게 노출하지 않음.
        "audit_metadata": {
            "audit_path_index":
                int(audit_path_index),
        },
    }
