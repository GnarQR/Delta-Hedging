import argparse
import json

from dotenv import load_dotenv

from systemone_client import (
    SystemOneHTTPClient,
)


load_dotenv()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "jev",
            "laya",
            "openjev",
        ],
    )

    args = parser.parse_args()

    client = (
        SystemOneHTTPClient.from_provider(
            args.provider
        )
    )

    result = client.choose(
        state={
            "message":
                "A hedge order was submitted at step 20 and filled at step 25."
        },
        question_id=
            "mechanism",
        instructions=(
            "Which mechanism best describes the state?"
        ),
        criteria={
            "execution_delay":
                "The submitted order is filled later.",
            "no_delay":
                "Submission and fill timing are aligned.",
        },
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
