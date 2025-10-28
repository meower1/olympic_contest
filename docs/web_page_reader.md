# WebPageReader

## Overview
`WebPageReader` is an HTTP-first tool that powers web-based reasoning for the contest agent. It can:
- Fetch raw content from URLs with redirect support.
- Execute scripted operations (`fetch`, `follow_link`, `submit_form`, `wait`) to mimic simple click flows without a full browser.
- Return both raw HTML/text and a cleaned plain-text representation.
- Provide structured traces for auditing API usage and staying within the contest's budget.

## Usage
```python
from solution import WebPageReader

reader = WebPageReader()
result = reader.fetch("https://en.wikipedia.org/wiki/Meow")

print(result.status_code)       # 200
print(result.url)               # Final URL after redirects
print(result.plain_text[:200])  # Human-readable text slice
```

### Executing Operations
```python
operations = [
    {"action": "follow_link", "text": "History"},
    {"action": "wait", "seconds": 1},
]
result = reader.execute("https://example.com", operations)
```

The `trace` attribute contains a chronological record of performed actions:
```python
for step in result.trace:
    print(step.action, step.url, step.status_code, step.note)
```
