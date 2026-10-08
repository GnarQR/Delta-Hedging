import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    try:
        from google import genai
    except ImportError:
        print(
            "google-genai is not installed."
        )
        print(
            "Run: pip install -U google-genai"
        )
        sys.exit(1)

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        print(
            "GEMINI_API_KEY is not set."
        )
        print(
            'PowerShell example: '
            '$env:GEMINI_API_KEY="YOUR_KEY"'
        )
        sys.exit(1)

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash-lite",
    )

    client = genai.Client(
        api_key=api_key
    )

    print(
        f"Testing Gemini model: {model}"
    )

    try:
        response = (
            client.models.generate_content(
                model=model,
                contents=(
                    "Return exactly the word OK."
                ),
            )
        )
    except Exception as exc:
        print(
            "\nGemini API request failed:"
        )
        print(
            type(exc).__name__,
            str(exc),
        )
        print(
            "\nIf this is a model-access error, "
            "try another Gemini model with:"
        )
        print(
            '$env:GEMINI_MODEL="gemini-3.5-flash-lite"'
        )
        sys.exit(2)

    print(
        "\nAPI response:"
    )
    print(
        response.text
    )

    print(
        "\nGemini API connection looks usable."
    )


if __name__ == "__main__":
    main()
