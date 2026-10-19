"""Build readable Jira exports from the saved, sanitized snapshot."""

import csv
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
snapshot = json.loads((ROOT / "jira-snapshot.json").read_text())


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def rich_text(value):
    if isinstance(value, dict):
        if "text" in value:
            return value["text"]
        return " ".join(rich_text(item) for item in value.get("content", []))
    if not value:
        return ""
    parser = PlainText()
    parser.feed(str(value))
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def text(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return value.get("value", value.get("name", json.dumps(value)))
    if isinstance(value, list):
        return "; ".join(str(text(item)) for item in value)
    return value


def custom(fields, label):
    return text(fields.get("customFields", {}).get(label, {}).get("value"))


columns = ["Key", "Title", "Type", "Status", "Assignee", "Parent", "Sprint",
           "Story points", "Effort estimate (hours)", "Actual logged hours",
           "Priority", "Probability", "Impact", "Mitigation", "Linked work",
           "Description", "Comments", "URL"]

for kind, issues in [("project-plan", snapshot["project_issues"]),
                     ("risk-plan", snapshot["risk_issues"]),
                     ("test-plan", snapshot["test_issues"])]:
    path = ROOT / f"{kind}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for issue in issues:
            f = issue["fields"]
            links = [f'{link["type"]["name"]}: '
                     f'{(link.get("inwardIssue") or link.get("outwardIssue"))["key"]}'
                     for link in f.get("issuelinks", [])]
            comments = "\n\n".join(rich_text(c.get("body", ""))
                                    for c in f.get("comment", {}).get("comments", []))
            writer.writerow([issue["key"], f["summary"], f["issuetype"]["name"],
                             f["status"]["name"], (f.get("assignee") or {}).get("displayName", ""),
                             f.get("parent", {}).get("key", ""), custom(f, "Sprint"),
                             custom(f, "Story point estimate"), custom(f, "Effort Estimate"),
                             f.get("timetracking", {}).get("timeSpentSeconds", 0) / 3600,
                             f.get("priority", {}).get("name", ""), custom(f, "Probability"),
                             custom(f, "Impact"), rich_text(custom(f, "Mitigation Strategy")),
                             "; ".join(links), rich_text(f.get("description")), comments,
                             f'https://basketball-strategy.atlassian.net/browse/{issue["key"]}'])
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == len(issues) + 1
    assert all(len(row) == len(columns) for row in rows)
    print(f"{path.name}: {len(issues)} items")
