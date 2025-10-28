# EntityCounter

## Overview
`EntityCounter` builds on `WebPageReader` to count occurrences of a target entity across one or more URLs. It supports:
- CSS selectors for HTML nodes.
- Regex-based text matching.
- Optional per-target scripted operations before counting (e.g., follow a link first).
- Aggregated totals plus per-URL breakdowns and action traces.

## Usage
```python
from solution import EntityCounter, WebPageReader

reader = WebPageReader()
counter = EntityCounter(reader)

result = counter.run(
    ["https://en.wikipedia.org/wiki/Meow"],
    "meow",
    spec={"regex": r"\bmeow\b"},
)

print(result["total"])          # 21
print(result["per_target"])     # {"https://en.wikipedia.org/wiki/Meow": 21}
```

### With Operations
```python
ops = [
    {"action": "follow_link", "selector": "a.more-details"},
]
result = counter.run(
    ["https://example.com/start"],
    "widget",
    spec={"selector": ".widget", "operations": ops},
)
```
