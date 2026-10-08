# 가상 시장을 임의로 생성

import numpy as np

# 위험중립 측도(Risk-neutral measure) 하에서의 주식 GBM(Geometric Brownian Motion) 경로를 생성하는 함수
# 금융공학에서는 주로 주가의 움직임이 GBM을 따른다고 가정하고 주식파생상품 가격을 결정함
# 주가의 drift가 일반적인 기대수익률 mu가 아니라 r-q로 설정됨
def generate_gbm_paths(
    S0=100.0,  # 초기 주가
    r=0.03,  # 무위험 이자율(Risk-free rate)
    sigma=0.20,  # 변동성(Volatility)
    T=1.0,  # 만기 / 전체 시뮬레이션 기간 (년 단위)
    n_steps=252,  # 시간 단계 수 (기간을 나누는 조각 수)
    n_paths=2000,  # 경로 수
    seed=42,  # 난수 생성기 (같은 주가 경로 생성을 위함)
    q=0.0,  # 연속 배당수익률(Continuous dividend yield)
):
    # 초기 주가 S0에서 출발하여, 무위험 이자율 r, 배당수익률 q, 변동성 sigma를 가진 위험중립 GBM 주가를
    # T년 동안 n_steps개 시점으로 나누어 n_paths개의 경로로 생성
    rng = np.random.default_rng(seed)
    dt = T / n_steps  # 한 step의 시간 간격
    z = rng.standard_normal((n_paths, n_steps))  # 각 경로 & 시간마다 Z ~ N(0,1) 랜덤 충격 생성

    log_ret = (  # GBM의 한 기간 로그 수익률
        (r - q - 0.5 * sigma**2) * dt  # drift term
        + sigma * np.sqrt(dt) * z  # random shock term
    )

    # log_price = np.cumsum(log_ret, axis=1)  # 시간에 따른 누적 로그수익률

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_ret, axis=1))

    return paths

# 생성된 주가 경로를 보고 실제로 관측된 변동성이 어느 정도였는가를 계산하는 함수
def realized_volatility(paths, T):
    """
    LLM agent에게 sigma_true를 알려주지 않고, paths를 가지고 계산한
    실제 변동성(realized volatility)을 증거로 제공하기 위해 사용하는 함수

    sigma_true : 시뮬레이터가 실제 시장을 생성할 때 사용한 숨겨진 진짜 값
    sigma_hedge : Hedger가 Delta를 계산할 때 사용하는 값 (시뮬레이터는 이 값을 알 수 없음)
    realized volatility : 생성된 주가 움직임을 보고 사후적으로 추정한 관측 변동성
    """
    log_ret = np.diff(np.log(paths), axis=1)  # 로그 수익률 계산
    n_steps = log_ret.shape[1]  
    dt = T / n_steps  # 한 step 시간 간격 계산

    # 각 주가 경로의 변동성을 계산
    rv = np.std(log_ret, axis=1, ddof=1) / np.sqrt(dt)  # 연율화된 변동성 계산 (sigma_annual=sigma_daily / sqrt(dt))

    # 모든 path의 realized volatility 평균을 반환
    return float(np.mean(rv)) 