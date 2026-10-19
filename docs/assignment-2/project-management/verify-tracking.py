"""Check the saved Jira evidence without changing Jira."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
snapshot = json.loads((ROOT / "jira-snapshot.json").read_text())
worklog_record = json.loads((ROOT / "actual-worklogs.json").read_text())
expected_worklogs = {entry["key"]: entry for entry in worklog_record["entries"]}
projects = {i["key"]: i for i in snapshot["project_issues"]}
risks = {i["key"]: i for i in snapshot["risk_issues"]}
tests = {i["key"]: i for i in snapshot["test_issues"]}
all_issues = {**projects, **risks, **tests}
checks = []


def check(name, passed, detail=None):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def linked(issue):
    return {(link.get("inwardIssue") or link.get("outwardIssue"))["key"]
            for link in issue["fields"].get("issuelinks", [])}


def status(key):
    return all_issues[key]["fields"]["status"]["name"]


def custom(issue, label):
    return issue["fields"].get("customFields", {}).get(label, {}).get("value")


cases = [i for i in tests.values() if i["fields"]["issuetype"]["id"] == "10012"]
suites = [i for i in tests.values() if i["fields"]["issuetype"]["id"] == "10013"]
check("Every test item uses the course's Task or Epic type", len(cases) + len(suites) == len(tests))
for issue in cases:
    parent = issue["fields"].get("parent", {}).get("key")
    check(f'{issue["key"]} belongs to an Epic test suite',
          parent in tests and tests[parent]["fields"]["issuetype"]["id"] == "10013", parent)
for issue in suites:
    check(f'{issue["key"]} uses Epic hierarchy', issue["fields"]["issuetype"].get("hierarchyLevel") == 1)

check("All 84 risks are present", len(risks) == 84)
for issue in risks.values():
    check(f'{issue["key"]} links to tracked work', any(k.startswith("SCRUM-") for k in linked(issue)))
    comments = issue["fields"].get("comment", {})
    check(f'{issue["key"]} comment history is complete',
          comments.get("total", 0) == len(comments.get("comments", [])))

fixes = {f"SCRUM-{n}" for n in range(538, 550)}
failed_cases = [i for i in cases if i["fields"]["status"]["name"] == "Fail"]
for issue in failed_cases:
    planned = [key for key in linked(issue) & fixes
               if key in projects and custom(projects[key], "Sprint")]
    check(f'{issue["key"]} has a linked repair with a planned sprint', bool(planned), planned)

for n in range(171, 179):
    key = f"BST-{n}"
    issue = tests[key]
    check(f"{key} is in the new export suite", issue["fields"].get("parent", {}).get("key") == "BST-170")
    check(f"{key} links to the implemented story", "SCRUM-486" in linked(issue))
    check(f"{key} has a current executed result", status(key) == "Pass")
    check(f"{key} does not retain a not-run description", "Not run yet" not in str(issue["fields"].get("description", "")))

check("Model Metrics regression passed", status("BST-145") == "Pass")
check("The filename failure remains in the history",
      any(c["id"] == "11212" for c in tests["BST-178"]["fields"]["comment"]["comments"]))
check("The filename regression links to its actual repair", "SCRUM-616" in linked(tests["BST-178"]))

for key in ["SCRUM-486", "SCRUM-542", "SCRUM-550", "SCRUM-616", "SCRUM-619",
            "SCRUM-604", "SCRUM-605", "SCRUM-620"]:
    check(f"{key} implementation is completed", status(key) == "Done")
check("The model publication is completed", status("SCRUM-603") == "Done" and status("SCRUM-606") == "Done")
check("Latency acceptance remains open", status("SCRUM-609") == "To Do" and status("BST-111") == "Verify")

for key in ["SCRUM-480", "SCRUM-526", "SCRUM-575", "SCRUM-486", "SCRUM-542",
            "SCRUM-550", "SCRUM-603", "SCRUM-609", "SCRUM-616"]:
    points = custom(projects[key], "Story point estimate")
    check(f"{key} has a point estimate", isinstance(points, (int, float)) and points > 0)

for key, seconds in {"SCRUM-440": 3600, "SCRUM-529": 14400, "SCRUM-530": 14400}.items():
    check(f"{key} retains its supplied Assignment 1 hours",
          projects[key]["fields"]["timetracking"].get("timeSpentSeconds") == seconds)
logged_total = 0
for key, entry in expected_worklogs.items():
    logs = projects[key]["fields"]["worklog"]["worklogs"]
    check(f"{key} has exactly its confirmed worklog", len(logs) == 1 and logs[0]["id"] == entry["worklog_id"])
    seconds = sum(log["timeSpentSeconds"] for log in logs)
    logged_total += seconds
    check(f"{key} matches the supplied hours", seconds == entry["seconds"])
    check(f"{key} uses the allocated work date", all(log["started"].startswith(entry["work_date"]) for log in logs))
check("Assignment 2 totals 13 hours without counting parent rollups twice", logged_total == 46800)
for n in range(600, 621):
    key = f"SCRUM-{n}"
    if key in projects:
        if key not in expected_worklogs:
            check(f"{key} has no additional actual worklog",
                  projects[key]["fields"].get("worklog", {}).get("total", 0) == 0)
        if projects[key]["fields"]["issuetype"].get("subtask"):
            hours = custom(projects[key], "Effort Estimate")
            check(f"{key} has a separate effort estimate", isinstance(hours, (int, float)) and hours > 0)

for n in range(588, 600):
    key = f"SCRUM-{n}"
    check(f"{key} duplicate is closed and linked", status(key) == "Done" and
          any(link["type"]["name"] == "Duplicate" for link in projects[key]["fields"].get("issuelinks", [])))

result = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "snapshot_captured_at_utc": snapshot["captured_at_utc"],
    "passed": all(c["passed"] for c in checks),
    "checks_passed": sum(c["passed"] for c in checks),
    "checks_total": len(checks),
    "test_cases": len(cases),
    "case_statuses": dict(Counter(i["fields"]["status"]["name"] for i in cases)),
    "test_suites": len(suites),
    "suite_statuses": dict(Counter(i["fields"]["status"]["name"] for i in suites)),
    "risk_statuses": dict(Counter(i["fields"]["status"]["name"] for i in risks.values())),
    "checks": checks,
}
(ROOT / "tracking-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
if not result["passed"]:
    raise SystemExit("Tracking verification failed; inspect tracking-verification.json.")
