from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solution import ContestantAgent, EntityCounter, WebPageReader


@dataclass
class TestResult:
    name: str
    success: bool
    detail: str


def run_test(name: str, func: Callable[[], None]) -> TestResult:
    try:
        func()
    except AssertionError as exc:
        return TestResult(name=name, success=False, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive path
        return TestResult(name=name, success=False, detail=f"Unexpected error: {exc}")
    return TestResult(name=name, success=True, detail="ok")


def test_fetch_basic() -> None:
    reader = WebPageReader()
    result = reader.fetch("https://en.wikipedia.org/wiki/Meow")
    assert result.status_code == 200, f"Unexpected status code {result.status_code}"
    assert "Meow" in result.plain_text, "Keyword 'Meow' not found in page content."


def test_follow_link_operation() -> None:
    reader = WebPageReader()
    operations = [{"action": "follow_link", "selector": "a"}]
    result = reader.execute("https://example.com", operations)
    assert "IANA" in result.plain_text, "Expected destination content not present after follow_link."


def test_entity_counter_word() -> None:
    reader = WebPageReader()
    counter = EntityCounter(reader)
    outcome = counter.run(
        ["https://en.wikipedia.org/wiki/Meow"],
        "meow",
        spec={"regex": r"\bmeow\b"},
    )
    expected = 21
    actual = outcome["total"]
    assert actual == expected, f"Expected {expected} occurrences, got {actual}."


def test_agent_handles_count_prompt() -> None:
    os.environ.setdefault("METIS_API_KEY", "test-key")
    agent = ContestantAgent(api_key=os.environ["METIS_API_KEY"])

    class _AssertingLLM:
        def __init__(self) -> None:
            self.invoked = False

        def complete(self, prompt: str) -> "_StubCompletion":
            self.invoked = True
            assert "Tool Observation" in prompt, "Tool context missing from prompt."
            assert "Total occurrences: 21" in prompt, "Incorrect tool data forwarded to LLM."
            return _StubCompletion("21")

    @dataclass
    class _StubCompletion:
        text: str

    llm = _AssertingLLM()
    agent.llm = llm  # type: ignore[assignment]

    prompt = (
        "How many times does the word 'meow' appear on the provided page?\n"
        "https://en.wikipedia.org/wiki/Meow\n"
    )
    answer = agent.solve_lock(prompt, history=[])
    assert answer == "21", f"Agent returned {answer}, expected '21'."
    assert llm.invoked, "LLM should be invoked for final answer generation."


def main() -> None:
    load_dotenv()
    tests = [
        ("fetch_basic", test_fetch_basic),
        ("follow_link_operation", test_follow_link_operation),
        ("entity_counter_word", test_entity_counter_word),
        ("agent_handles_count_prompt", test_agent_handles_count_prompt),
    ]

    results = [run_test(name, func) for name, func in tests]
    failures = [res for res in results if not res.success]

    for res in results:
        status = "PASS" if res.success else "FAIL"
        print(f"[{status}] {res.name}: {res.detail}")

    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed.")

    print("All web tool tests passed.")


if __name__ == "__main__":
    main()
