from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

load_dotenv()


class WebToolError(Exception):
    """Base exception for web tooling errors."""


class OperationFailed(WebToolError):
    """Raised when an interactive step cannot be completed."""


@dataclass
class OperationTrace:
    action: str
    url: str
    status_code: Optional[int]
    note: Optional[str] = None


@dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: Optional[str]
    raw_text: str
    plain_text: str
    soup: Optional[BeautifulSoup]
    headers: Mapping[str, str]
    bytes_downloaded: int
    trace: List[OperationTrace] = field(default_factory=list)


class WebPageReader:
    """
    HTTP-first web tool with optional scripted interactions and entity counting support.

    Operations are declarative dict objects with an ``action`` key. Supported actions:
      - ``fetch``: perform a direct HTTP request (GET by default).
      - ``follow_link``: locate an anchor via CSS selector or text and follow its href.
      - ``submit_form``: submit a form using provided field overrides.
      - ``wait``: no-op placeholder to align with prompts that expect waiting.
    """

    user_agent: str = "olympic-contest-agent/0.1"
    timeout: float = 30.0
    max_operations: int = 8

    def __init__(self) -> None:
        self._client_options: Dict[str, Any] = {
            "timeout": self.timeout,
            "headers": {"User-Agent": self.user_agent},
            "follow_redirects": True,
        }

    def run(
        self,
        url: str,
        operations: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> FetchResult:
        if operations:
            return self.execute(url, operations)
        return self.fetch(url)

    def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> FetchResult:
        method = method.upper()
        request_headers = dict(self._client_options["headers"])
        if headers:
            request_headers.update(headers)

        trace: List[OperationTrace] = []
        try:
            with httpx.Client(**self._client_options) as client:
                response = client.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=request_headers,
                )
        except httpx.RequestError as exc:
            trace.append(
                OperationTrace(
                    action=method,
                    url=url,
                    status_code=None,
                    note=f"request error: {exc}",
                )
            )
            raise WebToolError(f"HTTP request failed for {url}") from exc

        status = response.status_code
        content_type = response.headers.get("Content-Type")
        raw_text = response.text
        soup = None
        plain_text = raw_text

        if content_type and "html" in content_type.lower():
            soup = BeautifulSoup(raw_text, "html.parser")
            plain_text = soup.get_text(separator="\n", strip=True)
        elif content_type and "json" in content_type.lower():
            plain_text = response.text

        trace.append(
            OperationTrace(action=method, url=str(response.url), status_code=status)
        )

        return FetchResult(
            url=str(response.url),
            status_code=status,
            content_type=content_type,
            raw_text=raw_text,
            plain_text=plain_text,
            soup=soup,
            headers=dict(response.headers),
            bytes_downloaded=len(response.content),
            trace=trace,
        )

    def execute(
        self,
        url: str,
        operations: Sequence[Mapping[str, Any]],
    ) -> FetchResult:
        if len(operations) > self.max_operations:
            raise OperationFailed(
                f"Too many operations requested ({len(operations)} > {self.max_operations})."
            )

        result = self.fetch(url)
        trace = list(result.trace)

        for index, operation in enumerate(operations, start=1):
            action = operation.get("action")
            if not action:
                raise OperationFailed(f"Operation {index} missing 'action'.")

            action_lower = action.lower()
            if action_lower == "fetch":
                target_url = operation.get("url", result.url)
                method = operation.get("method", "GET")
                params = operation.get("params")
                data = operation.get("data")
                headers = operation.get("headers")
                result = self.fetch(target_url, method=method, params=params, data=data, headers=headers)
            elif action_lower == "follow_link":
                result = self._follow_link(result, operation)
            elif action_lower == "submit_form":
                result = self._submit_form(result, operation)
            elif action_lower == "wait":
                trace.append(
                    OperationTrace(
                        action="wait",
                        url=result.url,
                        status_code=result.status_code,
                        note=f"wait({operation.get('seconds', 0)})",
                    )
                )
                continue
            else:
                raise OperationFailed(f"Unsupported action '{action}' in operation {index}.")

            trace.append(
                OperationTrace(
                    action=action_lower,
                    url=result.url,
                    status_code=result.status_code,
                )
            )
            trace.extend(result.trace)

        result.trace = trace
        return result

    def _follow_link(self, current: FetchResult, operation: Mapping[str, Any]) -> FetchResult:
        if not current.soup:
            raise OperationFailed("Cannot follow link: current content is not HTML.")

        selector = operation.get("selector")
        link_text = operation.get("text")

        element = None
        if selector:
            element = current.soup.select_one(selector)
        elif link_text:
            link_text = str(link_text).strip().lower()
            element = current.soup.find(
                "a",
                string=lambda s: s and link_text in s.strip().lower(),
            )
        else:
            raise OperationFailed("follow_link requires 'selector' or 'text'.")

        if not element or not element.get("href"):
            raise OperationFailed("Unable to locate link for follow_link operation.")

        href = element.get("href")
        next_url = urljoin(current.url, href)
        return self.fetch(next_url)

    def _submit_form(self, current: FetchResult, operation: Mapping[str, Any]) -> FetchResult:
        if not current.soup:
            raise OperationFailed("Cannot submit form: current content is not HTML.")

        selector = operation.get("selector")
        form = None
        if selector:
            form = current.soup.select_one(selector)
        else:
            form = current.soup.find("form")

        if form is None:
            raise OperationFailed("submit_form could not locate the requested form.")

        form_method = str(operation.get("method") or form.get("method") or "GET").upper()
        form_action = operation.get("action") or form.get("action") or current.url
        submit_url = urljoin(current.url, form_action)

        payload: Dict[str, Any] = {}
        for input_el in form.select("input[name], textarea[name], select[name]"):
            name = input_el.get("name")
            if not name:
                continue

            if input_el.name == "select":
                option = input_el.find("option", selected=True)
                payload[name] = option.get("value") if option else ""
            else:
                payload[name] = input_el.get("value", "")

        overrides = operation.get("inputs") or {}
        payload.update(overrides)

        if form_method == "GET":
            return self.fetch(submit_url, method="GET", params=payload)
        return self.fetch(submit_url, method=form_method, data=payload)


class EntityCounter:
    """
    Count entities across one or more URLs or previously fetched documents.

    ``spec`` supports the following keys:
      - ``selector``: CSS selector for HTML documents.
      - ``regex``: custom regex pattern (overrides basic text matching).
      - ``case_sensitive``: whether simple text matching is case sensitive.
      - ``operations``: shared scripted operations to execute before counting.
    """

    def __init__(self, web_reader: WebPageReader) -> None:
        self.web_reader = web_reader

    def run(
        self,
        targets: Iterable[str],
        entity_name: str,
        *,
        spec: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = spec or {}
        selector = spec.get("selector")
        regex_pattern = spec.get("regex")
        case_sensitive = spec.get("case_sensitive", False)
        operations = spec.get("operations")

        compiled_regex: Optional[re.Pattern[str]] = None
        if regex_pattern:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled_regex = re.compile(regex_pattern, flags)

        per_target: Dict[str, int] = {}
        traces: Dict[str, List[Dict[str, Any]]] = {}

        for target in targets:
            ops_for_target: Optional[Sequence[Mapping[str, Any]]] = None
            if isinstance(operations, Mapping):
                ops_for_target = operations.get(target)
            elif isinstance(operations, Sequence):
                ops_for_target = operations

            if ops_for_target:
                result = self.web_reader.execute(target, ops_for_target)
            else:
                result = self.web_reader.fetch(target)

            count = self._count_in_result(
                result,
                entity_name,
                selector=selector,
                regex=compiled_regex,
                case_sensitive=case_sensitive,
            )
            per_target[result.url] = count
            traces[result.url] = [trace.__dict__ for trace in result.trace]

        total = sum(per_target.values())
        return {
            "entity": entity_name,
            "total": total,
            "per_target": per_target,
            "trace": traces,
        }

    def _count_in_result(
        self,
        result: FetchResult,
        entity_name: str,
        *,
        selector: Optional[str],
        regex: Optional[re.Pattern[str]],
        case_sensitive: bool,
    ) -> int:
        if selector and result.soup:
            elements = result.soup.select(selector)
            return len(elements)

        if regex:
            matches = regex.findall(result.plain_text)
            return len(matches)

        if not entity_name:
            return 0

        pattern = re.escape(entity_name)
        flags = 0 if case_sensitive else re.IGNORECASE
        matches = re.findall(rf"\b{pattern}\b", result.plain_text, flags)
        return len(matches)


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
    """Return concrete instances for the Phase I toolchain."""
    web_reader = WebPageReader()
    return {
        "web_reader": web_reader,
        "entity_counter": EntityCounter(web_reader),
        "csv_analyzer": CSVAnalyzer(),
        "api_client": APIRequestTool(),
        "calculator": PersianAwareCalculator(),
    }


class ContestantAgent:
    """
    Baseline agent template for the Olympic NLP Agents contest.

    This scaffold wires up a LlamaIndex OpenAI-like LLM with configurable tooling.
    It includes a rule-based fast path for common counting prompts so the agent can
    respond deterministically when the task is covered by available tools.
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
        prompt_text = self._extract_prompt(problem)

        tool_context = self._gather_tool_context(prompt_text)

        prompt = self._construct_prompt(prompt_text, history, tool_context)

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
        prompt_text: str,
        history: List[Tuple[str, str]],
        tool_context: Optional[str],
    ) -> str:
        formatted_history = self._format_history(history)

        sections = [
            f"System: {self.system_prompt}",
            formatted_history,
            tool_context or "",
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

    def _gather_tool_context(self, prompt_text: str) -> Optional[str]:
        urls = self._extract_urls(prompt_text)
        if not urls:
            return None

        lowered = prompt_text.lower()
        if "how many" not in lowered and "count" not in lowered:
            return None

        entity = self._extract_entity_name(prompt_text)
        if not entity:
            return None

        counter_obj = self.tools.get("entity_counter")
        if not isinstance(counter_obj, EntityCounter):
            return None

        spec = {"regex": rf"\b{re.escape(entity)}\b"}

        try:
            result = counter_obj.run(urls, entity, spec=spec)
        except WebToolError:
            return None
        except Exception:
            return None

        total = result.get("total")
        if total is None:
            return None

        per_target = result.get("per_target", {})
        details = ", ".join(f"{url}: {count}" for url, count in per_target.items())

        context_lines = [
            "Tool Observation:",
            f"- Entity counted: {entity}",
            f"- Total occurrences: {total}",
        ]
        if details:
            context_lines.append(f"- Per target: {details}")

        return "\n".join(context_lines)

    @staticmethod
    def _extract_urls(prompt_text: str) -> List[str]:
        url_pattern = re.compile(r"https?://[^\s)]+")
        urls = []
        for match in url_pattern.finditer(prompt_text):
            url = match.group(0).rstrip(').,;')
            urls.append(url)
        return urls

    @staticmethod
    def _extract_entity_name(prompt_text: str) -> Optional[str]:
        for pattern in (r"'([^']+)'", r'"([^"]+)"'):
            match = re.search(pattern, prompt_text)
            if match:
                candidate = match.group(1).strip()
                if candidate and not candidate.startswith("http"):
                    return candidate

        word_match = re.search(
            r"(?:count|count of|how many(?: times)? does(?: the)? word)\s+([A-Za-z0-9_\-]+)",
            prompt_text,
            flags=re.IGNORECASE,
        )
        if word_match:
            candidate = word_match.group(1).strip()
            if candidate:
                return candidate

        return None


__all__ = ["ContestantAgent"]
