from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

load_dotenv()


@dataclass
class WebPageReader:
    """
    Placeholder for a tool that fetches and optionally interacts with web pages.

    Responsibilities for the real implementation:
    - Download raw text or structured content from a given URL.
    - Execute simple scripted operations (e.g., button click) if requested.
    - Return normalized text that the agent can reason over.
    """

    user_agent: str = "olympic-contest-agent/0.1"

    def run(self, url: str, operations: Iterable[str] | None = None) -> str:
        raise NotImplementedError("WebPageReader.run is pending implementation.")


@dataclass
class EntityCounter:
    """
    Placeholder tool to count specific entities within one or more resources.

    The future implementation should:
    - Re-use the web reader/tooling substrate for downloading content.
    - Support structured selectors to identify the target entity type.
    - Return integer counts or raise meaningful errors when the entity is absent.
    """

    def run(self, targets: Iterable[str], entity_name: str) -> int:
        raise NotImplementedError("EntityCounter.run is pending implementation.")


@dataclass
class CSVAnalyzer:
    """
    Placeholder tool dedicated to inspecting CSV datasets.

    The complete implementation should:
    - Download CSV content from remote URLs.
    - Provide high-level summaries (headers, row count, sample rows).
    - Answer targeted questions about records, filtered by field names and IDs.
    - Enforce token-aware truncation when returning data to the LLM.
    """

    max_preview_rows: int = 5

    def summarize(self, source: str) -> Mapping[str, Any]:
        raise NotImplementedError("CSVAnalyzer.summarize is pending implementation.")

    def lookup(self, source: str, field: str, values: Iterable[str]) -> Mapping[str, Any]:
        raise NotImplementedError("CSVAnalyzer.lookup is pending implementation.")


@dataclass
class APIRequestTool:
    """
    Placeholder tool to perform authenticated HTTP API calls.

    The final version should:
    - Support multiple HTTP verbs with retries and timeout controls.
    - Build requests from LLM-provided parameters in a safe manner.
    - Return structured responses or targeted fields for downstream reasoning.
    """

    timeout: float = 30.0

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError("APIRequestTool.request is pending implementation.")


@dataclass
class PersianAwareCalculator:
    """
    Placeholder calculator that will convert Persian textual math into numeric results.

    Implementation goals:
    - Map Persian numerals and operators to arithmetic expressions.
    - Evaluate safely without exposing `eval`.
    - Provide helpful error messages when parsing fails.
    """

    def evaluate(self, expression: str) -> Any:
        raise NotImplementedError("PersianAwareCalculator.evaluate is pending implementation.")


def build_default_toolkit() -> Dict[str, Any]:
    """Return placeholder instances for the Phase I toolchain."""
    return {
        "web_reader": WebPageReader(),
        "entity_counter": EntityCounter(),
        "csv_analyzer": CSVAnalyzer(),
        "api_client": APIRequestTool(),
        "calculator": PersianAwareCalculator(),
    }


class ContestantAgent:
    """
    Baseline agent template for the Olympic NLP Agents contest.

    This scaffold wires up a LlamaIndex OpenAI-like LLM with configurable tooling.
    The heavy lifting for prompt parsing, history reasoning, and tool usage will be
    implemented iteratively during Phase I.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key or os.getenv("METIS_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required to initialize ContestantAgent.")

        self.base_url = os.getenv("LLM_BASE_URL", "https://api.metisai.ir/openai/v1")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")

        # Configure the OpenAI-like LLM as described in the contest brief.
        self.llm = OpenAILike(
            model=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key,
            timeout=60,
            temperature=0.0,
            is_chat_model=True,
        )
        Settings.llm = self.llm

        self.system_prompt = (
            "You are an Olympic NLP contest agent. "
            "Return only the final answer required by the lock prompt. "
            "Use tools conservatively; every external call consumes budget. "
            "If insufficient data is available, clearly request guidance."
        )

        # Placeholder toolkit wiring; each tool lives in its own module.
        self.tools = build_default_toolkit()

    def solve_lock(
        self,
        problem: Dict[str, Any] | str,
        history: List[Tuple[str, str]],
    ) -> str:
        """
        Solve a single Phase I lock.

        Args:
            problem: Either a raw string prompt or a dict payload from the judge.
            history: List of (prompt, response) tuples representing past attempts.
        """
        prompt = self._construct_prompt(problem, history)

        try:
            completion = self.llm.complete(prompt)
        except Exception as exc:  # pragma: no cover - defensive path
            return f"Agent execution failed: {exc}"

        return completion.text.strip()

    def choose_path(self, scenario_prompt: str) -> List[str]:
        """
        Decide the path for Phase II maze scenarios.

        For now this is a stub returning a placeholder path.
        """
        # TODO: Implement a lightweight decision model for maze scenarios.
        return ["Path_A"]

    def _construct_prompt(
        self,
        problem: Dict[str, Any] | str,
        history: List[Tuple[str, str]],
    ) -> str:
        prompt_text = self._extract_prompt(problem)
        formatted_history = self._format_history(history)

        sections = [
            f"System: {self.system_prompt}",
            formatted_history,
            "Current Lock Prompt:",
            prompt_text,
            "Required Output: Provide only the final answer as a single string.",
        ]

        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _extract_prompt(problem: Dict[str, Any] | str) -> str:
        if isinstance(problem, str):
            return problem.strip()

        for key in ("prompt", "description", "text"):
            if key in problem and isinstance(problem[key], str):
                return problem[key].strip()

        return str(problem)

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ""

        lines = ["Conversation History:"]
        for idx, (attempt_prompt, attempt_answer) in enumerate(history, start=1):
            lines.append(f"[Attempt {idx}] Prompt: {attempt_prompt}")
            lines.append(f"[Attempt {idx}] Answer: {attempt_answer}")
        return "\n".join(lines)


__all__ = ["ContestantAgent"]
