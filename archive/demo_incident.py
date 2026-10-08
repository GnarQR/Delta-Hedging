# 장애가 발생했을 때 진단 Agent가 원인을 찾아가는 환경을 만들고 시험
from pprint import pprint
from incident_env import HedgingIncidentEnv  # 가상 장애 환경을 가져옴

def main():
    # 숨겨진 Incident를 가진 환경을 생성
    env = HedgingIncidentEnv(
        incident_type="vol_misspec",  # 실제 장애 원인 (LLM에게 보여주지 않음)
        seed=11,
        n_paths=3000,
    )

    print("\n=== 1. Initial observation ===")
    pprint(env.initial_observation())  # Agent가 장애를 처음 발견했을 때 볼 수 있는 초기 증상

    print("\n=== 2. Available diagnostic tools ===")
    pprint(env.available_tools())  # Agent가 사용할 수 있는 진단 도구 목록

    # Below is a HAND-WRITTEN diagnostic sequence just to demonstrate
    # what an LLM agent would later do interactively.
    print("\n=== 3. Agent queries market evidence ===")
    pprint(env.inspect_market())  # 실제 시장을 조사

    print("\n=== 4. Agent queries hedger configuration ===")
    pprint(env.inspect_model_config())  # Hedger가 어떤 설정을 사용하고 있는지 확인

    print("\n=== 5. Agent tests a hypothesis with a counterfactual ===")
    pprint(env.run_counterfactual("use_realized_volatility"))  # Counterfactual 실험

    print("\n=== 6. Evaluator-only ground truth ===")
    pprint(env.evaluator_ground_truth())  # Agent가 실제 원인을 제대로 찾았는지 평가용

    print(
        "\nNext step: replace the hand-written query sequence with an LLM "
        "that chooses which tool to call and when to stop."
    )


if __name__ == "__main__":
    main()
