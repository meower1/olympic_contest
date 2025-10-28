from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solution import ContestantAgent

PROMPT = """how many rows are there in the provided CSV file?
https://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a851711e20c5eb28f9d54742d1fe2dc/datasets.csv
"""


def main() -> None:
    load_dotenv()

    api_key = os.getenv("METIS_API_KEY")
    if not api_key or api_key == "CHANGE_ME":
        raise RuntimeError(
            "Set METIS_API_KEY in the environment or .env before running."
        )

    agent = ContestantAgent(api_key=api_key)
    result = agent.solve_lock(PROMPT, history=[])

    print("Agent answer:", result)


if __name__ == "__main__":
    main()
