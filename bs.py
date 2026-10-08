# GBM 함수가 만든 주가 St를 받아 그 주가를 활용하는 함수
import numpy as np
from scipy.stats import norm

# Black-Scholes 유럽형 콜옵션 가격을 계산하는 함수
# Call option : 만기 때 주식을 행사가격 K에 살 수 있는 권리
# K = 100인 콜옵션은 만기 때 주가가 100보다 높으면 max(St-100,0)만큼 payoff
def bs_call_price(S, K, tau, r, sigma, q=0.0):
    """
    S : 현재 주가, K : 행사가격, tau : 현재부터 만기까지 남은 시간 (0.5 = 6개월)
    r : 무위험 이자율 (이론적으로 전혀 위험이 없는 투자에서 얻을 수 있는 수익률, ex) 국채, 정부 채권 등)
    sigma : 연율화된 변동성(annualized volatility), q : 연속 배당수익률
    """
    S = np.asarray(S, dtype=float)  # 여러 주가를 한 번에 계산하기 위해 Numpy array로 변환
    tau = np.asarray(tau, dtype=float)

    payoff = np.maximum(S - K, 0.0)  # 콜옵션 만기 payoff : max(St-K,0)
    safe_tau = np.maximum(tau, 1e-12)
    sqrt_tau = np.sqrt(safe_tau)  # Black-Scholes 공식에서 d1 계산 시 0으로 나누는 것 방지

    d1 = (
        np.log(np.maximum(S, 1e-300) / K)
        + (r - q + 0.5 * sigma**2) * safe_tau
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau 

    price = (
        S * np.exp(-q * safe_tau) * norm.cdf(d1)
        - K * np.exp(-r * safe_tau) * norm.cdf(d2)
    )  # Black-Scholes 공식에 따른 콜옵션 가격 계산

    return np.where(tau <= 1e-12, payoff, price)  # 만기라면 payoff 반환, 아니면 계산된 가격 반환

# Black-Scholes 콜옵션의 델타(Delta)를 계산하는 함수
# Delta = dC/dS, 즉 주가가 1만큼 변할 때 옵션 가격이 대략 얼마나 변하는가
# If Delta = 0.6, 주가가 1 올라가면 콜옵션 가격이 대략 0.6 상승 (dC=0.6dS)
# 옵션을 매도했다면 Delta값이 음수 -> 그만큼 주식을 매수해 Delta Hedging 가능
def bs_call_delta(S, K, tau, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)

    safe_tau = np.maximum(tau, 1e-12)
    sqrt_tau = np.sqrt(safe_tau)

    d1 = (
        np.log(np.maximum(S, 1e-300) / K)
        + (r - q + 0.5 * sigma**2) * safe_tau
    ) / (sigma * sqrt_tau)

    delta = np.exp(-q * safe_tau) * norm.cdf(d1)  # Black-Scholes 공식에 따른 콜옵션 델타 계산

    # 만기 처리 : 콜옵션 가치 Ct=max(St-K,0)이므로
    # 미분하면 delta = 0 if St < K, delta = 1 if St > K
    # St = K일 경우 미분이 정의되지 않아 편의상 0.5 사용
    maturity_delta = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))

    return np.where(tau <= 1e-12, maturity_delta, delta)
