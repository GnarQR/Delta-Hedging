import os
import time

import requests
from dotenv import load_dotenv


load_dotenv()


class SystemOneHTTPClient:
    """
    Generic client for the Jev-compatible /v1/systemone protocol.

    Works with:
    - TypeSafe Jev
    - Laya's Jev-compatible server
    - OpenJev implementations that expose the same wire format
    """

    def __init__(
        self,
        base_url,
        model=None,
        api_key=None,
        timeout=60,
        max_retries=3,
        provider_name="systemone",
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = None if model is None else str(model)
        self.api_key = api_key or None
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.provider_name = str(provider_name)

    @classmethod
    def from_provider(cls, provider):
        provider = str(provider).lower()

        if provider == "jev":
            return cls(
                base_url=os.getenv(
                    "JEV_BASE_URL",
                    "https://api.typesafe.ai",
                ),
                model=os.getenv(
                    "JEV_MODEL",
                    "jev-latest",
                ),
                api_key=os.getenv(
                    "TYPESAFE_API_KEY",
                ),
                provider_name="jev",
            )

        if provider == "laya":
            return cls(
                base_url=os.getenv(
                    "LAYA_BASE_URL",
                    "http://127.0.0.1:8000",
                ),
                model=os.getenv(
                    "LAYA_MODEL",
                    "laya",
                ),
                api_key=os.getenv(
                    "LAYA_API_KEY",
                ),
                provider_name="laya",
            )

        if provider == "openjev":
            return cls(
                base_url=os.getenv(
                    "OPENJEV_BASE_URL",
                    "http://127.0.0.1:3000",
                ),
                model=os.getenv(
                    "OPENJEV_MODEL",
                    "openjev",
                ),
                api_key=os.getenv(
                    "OPENJEV_API_KEY",
                ),
                provider_name="openjev",
            )

        raise ValueError(
            f"Unknown System One provider: {provider}"
        )

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers[
                "Authorization"
            ] = f"Bearer {self.api_key}"

        return headers

    def choose(
        self,
        *,
        state,
        question_id,
        instructions,
        criteria,
    ):
        """
        Submit one Choice question and return normalized result:

        {
          "choice": "...",
          "probabilities": {...},
          "confidence": ...,
          "raw": {...}
        }
        """
        payload = {
            "state": state,
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }

        # TypeSafe requires model; some local servers accept it optionally.
        if self.model:
            payload["model"] = self.model

        url = (
            self.base_url
            + "/v1/systemone"
        )

        last_exc = None

        for attempt in range(
            self.max_retries
        ):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code in {
                    429,
                    503,
                    529,
                }:
                    wait = min(
                        2 ** attempt,
                        8,
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                raw = response.json()

                answer = raw[
                    "answers"
                ][
                    question_id
                ]

                probabilities = {
                    str(k): float(v)
                    for k, v
                    in answer.get(
                        "probabilities",
                        {},
                    ).items()
                }

                choice = str(
                    answer[
                        "choice"
                    ]
                )

                # If a clone omits probabilities, preserve a usable one-hot
                # fallback but mark confidence as unavailable.
                if not probabilities:
                    probabilities = {
                        key:
                            1.0
                            if key == choice
                            else 0.0
                        for key
                        in criteria
                    }

                return {
                    "provider":
                        self.provider_name,
                    "resolved_model":
                        raw.get(
                            "model",
                            self.model,
                        ),
                    "choice":
                        choice,
                    "probabilities":
                        probabilities,
                    "confidence":
                        answer.get(
                            "confidence"
                        ),
                    "usage":
                        raw.get(
                            "usage"
                        ),
                    "raw":
                        raw,
                }

            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            f"System One request failed after "
            f"{self.max_retries} attempts: {last_exc}"
        )
