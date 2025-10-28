from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    Inspect CSV datasets and provide lightweight analytics.

    The implementation intentionally keeps the heuristics simple so the
    surrounding agent can prompt for additional logic when required.
    """

    max_preview_rows: int = 5
    timeout: float = 30.0
    user_agent: str = "olympic-contest-csv-analyzer/0.1"

    def summarize(self, source: str) -> Mapping[str, Any]:
        """Return headers, row count, and a lightweight sample preview."""
        try:
            data = self._fetch_csv(source)
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": f"Failed to load CSV: {exc}"}

        if not data:
            return {
                "source": source,
                "headers": [],
                "total_rows": 0,
                "sample_row": None,
                "preview": [],
            }

        headers = list(data[0].keys())
        preview = data[: self.max_preview_rows]

        return {
            "source": source,
            "headers": headers,
            "total_rows": len(data),
            "sample_row": data[0],
            "preview": preview,
        }

    def lookup(self, source: str, field: str, values: Iterable[str]) -> Mapping[str, Any]:
        """Find rows where ``field`` matches one of ``values``."""
        try:
            data = self._fetch_csv(source)
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": f"Failed to load CSV: {exc}"}

        if not data:
            return {"error": "CSV file is empty or could not be loaded."}

        headers = list(data[0].keys())
        if field not in headers:
            return {
                "error": f"Field '{field}' not found in CSV headers.",
                "headers": headers,
            }

        lookup_values: List[str]
        if values is None:
            lookup_values = []
        elif isinstance(values, str):
            lookup_values = [values]
        else:
            lookup_values = list(values)

        normalized_values = {str(v) for v in lookup_values if v is not None}
        matches = [row for row in data if str(row.get(field)) in normalized_values]
        found_values = {str(row.get(field)) for row in matches}
        missing_values = sorted(normalized_values - found_values)

        preview = matches[: self.max_preview_rows]

        return {
            "source": source,
            "field": field,
            "requested_values": sorted(normalized_values),
            "total_matches": len(matches),
            "preview": preview,
            "missing_values": missing_values,
        }

    def query(self, source: str, question: str) -> str:
        """
        Answer general questions about the CSV data.

        Args:
            source: URL of the CSV file
            question: Natural language question about the data

        Returns:
            String answer to the question
        """
        try:
            data = self._fetch_csv(source)

            if not data:
                return "CSV file is empty or could not be loaded."

            # Prepare context for answering
            headers = list(data[0].keys())
            row_count = len(data)

            # This is a simple implementation - in production you might want
            # to use the LLM to interpret the question and generate appropriate code
            question_lower = question.lower()

            # Handle common question patterns (English)
            if "how many" in question_lower or "count" in question_lower:
                if "column" in question_lower or "field" in question_lower:
                    return f"The CSV has {len(headers)} columns: {', '.join(headers)}"
                elif "row" in question_lower or "record" in question_lower:
                    return f"The CSV has {row_count} rows."

            if "what are" in question_lower and ("column" in question_lower or "field" in question_lower):
                return f"Columns: {', '.join(headers)}"

            if "first" in question_lower or "sample" in question_lower:
                sample = data[:3]
                return f"Sample rows:\n" + "\n".join([str(row) for row in sample])

            # Handle a minimal set of Persian prompts.
            persian_question = question.strip()
            if persian_question and ("چند" in persian_question and ("ردیف" in persian_question or "ستون" in persian_question)):
                return (
                    f"فایل دارای {row_count} ردیف و {len(headers)} ستون است. "
                    f"ستون‌ها عبارتند از: {', '.join(headers)}"
                )

            # Default response with summary
            return (
                f"CSV Summary:\n"
                f"- Total rows: {row_count}\n"
                f"- Columns: {', '.join(headers)}\n"
                f"- Sample data: {data[0]}"
            )
        except Exception as e:  # pragma: no cover - defensive
            return f"Failed to answer question: {str(e)}"

    def run(self, **kwargs) -> Any:
        """
        Generic run method to handle different operations.

        Supported operations:
        - summarize: Get summary of CSV
        - lookup: Find records by field value
        - query: Answer natural language questions
        """
        operation = kwargs.get("operation", "summarize")
        source = kwargs.get("source") or kwargs.get("url")

        if not source:
            return {"error": "Source URL is required"}

        if operation == "summarize":
            return self.summarize(source)
        elif operation == "lookup":
            field = kwargs.get("field", "id")
            values = kwargs.get("values", [])
            return self.lookup(source, field, values)
        elif operation == "query":
            question = kwargs.get("question", "")
            return self.query(source, question)
        else:
            return {"error": f"Unknown operation: {operation}"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_csv(self, source: str) -> List[Dict[str, str]]:
        """Download or read CSV content and return it as dictionaries."""
        if not source:
            raise ValueError("CSV source must be provided.")

        if source.startswith("http://") or source.startswith("https://"):
            headers = {"User-Agent": self.user_agent}
            try:
                with httpx.Client(timeout=self.timeout, headers=headers) as client:
                    response = client.get(source)
                    response.raise_for_status()
                    text = response.text
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Failed to fetch remote CSV: {exc}") from exc
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found at {source}")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")

        buffer = io.StringIO(text)
        reader = csv.DictReader(buffer)
        data = [dict(row) for row in reader]
        return data


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
        input_files = self._extract_input_files(problem)

        fast_answer = self._try_answer_with_csv_tool(prompt_text, input_files)
        if fast_answer:
            return fast_answer

        tool_context = self._gather_tool_context(prompt_text, input_files)

        prompt = self._construct_prompt(prompt_text, history, tool_context, input_files)

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
        input_files: Sequence[Mapping[str, Any]],
    ) -> str:
        formatted_history = self._format_history(history)
        files_section = self._format_input_files(input_files)

        sections = [
            f"System: {self.system_prompt}",
            formatted_history,
            tool_context or "",
            files_section or "",
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
    def _extract_input_files(problem: Dict[str, Any] | str) -> List[Mapping[str, Any]]:
        if isinstance(problem, Mapping):
            files = problem.get("input_files")
            if isinstance(files, Sequence):
                result: List[Mapping[str, Any]] = []
                for file in files:
                    if isinstance(file, Mapping):
                        result.append(file)
                return result
        return []

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ""

        lines = ["Conversation History:"]
        for idx, (attempt_prompt, attempt_answer) in enumerate(history, start=1):
            lines.append(f"[Attempt {idx}] Prompt: {attempt_prompt}")
            lines.append(f"[Attempt {idx}] Answer: {attempt_answer}")
        return "\n".join(lines)

    @staticmethod
    def _format_input_files(input_files: Sequence[Mapping[str, Any]]) -> Optional[str]:
        if not input_files:
            return None

        lines = ["Attached Files:"]
        for file_info in input_files:
            name = str(file_info.get("filename") or file_info.get("name") or "unnamed")
            file_type = str(file_info.get("type") or "unknown")
            url = file_info.get("url") or file_info.get("source")
            description = file_info.get("description")

            parts = [f"{name} ({file_type})"]
            if description:
                parts.append(str(description))
            if url:
                parts.append(str(url))

            lines.append("- " + " – ".join(parts))

        return "\n".join(lines)

    def _gather_tool_context(
        self,
        prompt_text: str,
        input_files: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        urls = self._extract_urls(prompt_text)
        for file_info in input_files:
            url = file_info.get("url") or file_info.get("source")
            if url:
                urls.append(str(url))

        if urls:
            urls = list(dict.fromkeys(urls))

        if not urls:
            return None

        contexts: List[str] = []

        csv_context = self._build_csv_context(prompt_text, urls, input_files)
        if csv_context:
            contexts.append(csv_context)

        entity_context = self._build_entity_context(prompt_text, urls)
        if entity_context:
            contexts.append(entity_context)

        if contexts:
            return "\n\n".join(contexts)
        return None

    def _try_answer_with_csv_tool(
        self,
        prompt_text: str,
        input_files: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        csv_tool = self.tools.get("csv_analyzer")
        if not isinstance(csv_tool, CSVAnalyzer):
            return None

        urls = self._extract_urls(prompt_text)
        csv_urls = [url for url in urls if self._is_csv_url(url)]
        for file_info in input_files:
            url = file_info.get("url") or file_info.get("source")
            if url and self._is_csv_url(str(url)):
                csv_urls.append(str(url))
        if not csv_urls:
            return None

        lowered = prompt_text.lower()
        row_keywords = ("row", "rows", "record", "records")
        column_keywords = ("column", "columns", "field", "fields", "header", "headers")
        persian_terms = ("ردیف", "ستون")
        asks_rows = any(token in lowered for token in row_keywords) or "چند" in prompt_text and "ردیف" in prompt_text
        asks_columns = any(token in lowered for token in column_keywords) or "چند" in prompt_text and "ستون" in prompt_text
        wants_headers = (
            "header" in lowered
            or "headers" in lowered
            or ("column" in lowered and ("list" in lowered or "name" in lowered or "names" in lowered))
            or ("ستون" in prompt_text and "نام" in prompt_text)
        )
        wants_sample = "sample" in lowered or "first" in lowered

        if not (asks_rows or asks_columns or wants_headers or wants_sample):
            return None

        source = csv_urls[0]
        summary = csv_tool.summarize(source)
        if not isinstance(summary, Mapping) or summary.get("error"):
            return None

        headers = summary.get("headers", [])
        row_count = summary.get("total_rows")
        sample_row = summary.get("sample_row")
        col_count = len(headers)

        if wants_headers and headers:
            return ", ".join(headers)

        if asks_rows and asks_columns and "چند" in prompt_text:
            return (
                f"فایل دارای {row_count} ردیف و {col_count} ستون است. "
                f"ستون‌ها عبارتند از: {', '.join(headers)}"
            )

        if asks_rows and not asks_columns and row_count is not None:
            if "چند" in prompt_text:
                return f"فایل دارای {row_count} ردیف است."
            return str(row_count)

        if asks_columns and not asks_rows:
            if "چند" in prompt_text:
                return f"فایل دارای {col_count} ستون است."
            return str(col_count)

        if asks_rows and asks_columns:
            if "csv" in lowered or "file" in lowered:
                return f"rows={row_count}, columns={col_count}"
            return f"{row_count} rows, {col_count} columns"

        if wants_sample and sample_row:
            return str(sample_row)

        answer = csv_tool.query(source, prompt_text)
        if answer.startswith("Failed"):
            return None
        return answer

    def _build_csv_context(
        self,
        prompt_text: str,
        urls: Sequence[str],
        input_files: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        csv_tool = self.tools.get("csv_analyzer")
        if not isinstance(csv_tool, CSVAnalyzer):
            return None

        csv_urls = [url for url in urls if self._is_csv_url(url)]
        if not csv_urls:
            return None

        lowered = prompt_text.lower()
        triggers = ("csv", "row", "rows", "column", "columns", "header", "headers", "sample")
        persian_triggers = ("ردیف", "ستون")
        has_trigger = any(trigger in lowered for trigger in triggers) or any(
            trig in prompt_text for trig in persian_triggers
        )
        csv_from_files = any(
            self._is_csv_url(str(file_info.get("url") or file_info.get("source")))
            for file_info in input_files
        )
        if not has_trigger and not csv_from_files:
            return None

        summaries: List[str] = []
        file_lookup: Dict[str, Mapping[str, Any]] = {}
        for file_info in input_files:
            url = file_info.get("url") or file_info.get("source")
            if url:
                file_lookup[str(url)] = file_info

        for source in csv_urls[:2]:  # limit to first two sources for brevity
            summary = csv_tool.summarize(source)
            if not isinstance(summary, Mapping):
                continue
            if summary.get("error"):
                continue
            headers = summary.get("headers", [])
            total_rows = summary.get("total_rows")
            sample_row = summary.get("sample_row")
            label = file_lookup.get(source, {}).get("filename")
            if label:
                header_line = f"- File: {label}"
            else:
                header_line = f"- Source: {source}"
            lines = [
                "Tool Observation (CSV Analyzer):",
                header_line,
                f"- URL: {source}",
                f"- Headers: {headers}",
                f"- Total rows: {total_rows}",
            ]
            if sample_row:
                lines.append(f"- Sample row: {sample_row}")
            summaries.append("\n".join(lines))

        if summaries:
            return "\n\n".join(summaries)
        return None

    def _build_entity_context(self, prompt_text: str, urls: Sequence[str]) -> Optional[str]:
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
    def _is_csv_url(url: str) -> bool:
        normalized = url.lower().split("?", 1)[0]
        return normalized.endswith(".csv")

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
