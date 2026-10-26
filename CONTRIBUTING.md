# Project work checklist

Use this checklist for each sprint deliverable. Keep the Jira work item, shared Visual Paradigm model, code and test results connected so either team member can follow the change.

## Before implementation

1. Confirm the owner and sprint. Give the parent work item a story-point estimate and split the work into specific Dev Subtasks with effort estimates and acceptance criteria. Record actual individual time separately and count it once.
2. Put the exact VP use-case name and stable model ID in Jira's Parent Use Case field. Repairs can refer to an existing use case. For deployment, recovery or project-management work, identify the affected use cases and the relevant model or operating procedure.
3. In VP, place each new use case in the appropriate functional overview. Write its success steps and relevant error paths in **Use Case Details → Flow of Events**. Include the ticket's acceptance criteria there.
4. Check the use case's collaboration overview, structural design and interaction diagrams. Name lifelines consistently and label calls and returns with their parameters and result types. Update deployment details when communication or hosting changes.
5. Connect the screen behavior to the UI wireflow. Show where users enter the feature, its result, and relevant loading, empty and error states. Offline jobs should have an explicit entry point and output instead of an invented screen.

## During implementation

- Follow the agreed use-case flow and design. If the implementation needs a design change, update both.
- Keep displayed results and downloads tied to the same inputs and data version. Clear or clearly invalidate results when those inputs change.
- Add tests for the acceptance criteria, relevant access rules and failure paths. Keep the existing test history.
- Integrate into the shared team repository. Fetch and inspect teammate changes before pushing. Commit coherent, substantive changes with short descriptions; use the issue key when it helps locate the work.
- Publish substantive VP changes to the shared project. Use a short comment describing the change and related issue. The repository and VPository account histories identify the contributor.

## Before marking work complete

- Run the relevant test-plan cases and record the actual result, tested release and evidence. Put Test Cases under Epic-level Test Suites and keep them visible in the Kanban board columns.
- Link each failed case to a fix ticket with a priority and planned sprint. Keep the original failure when a later run passes.
- Review the risks affected by the change. Link the mitigation work and add a specific update explaining what changed, what was checked and what remains. Repeated generic comments do not show monitoring.
- Verify the shared VP revision contains the intended changes. A local save is not a published revision.
- Add a short Jira completion comment explaining what changed and how it was checked. Link the commit, test cases and model reference. Attach an app screenshot when it shows useful evidence.
- Mark only completed work Done. Keep outstanding model publication, acceptance checks and future requirements visible in their own work items.

## Reviewing earlier work

Fill missing links and tracking from the actual source, model and test evidence. Preserve original ownership, commits, execution dates, results and work logs. New estimates describe the current scope; they do not replace actual time or recreate earlier planning history.

Use the [current release guide](docs/iteration-17/README.md) for verified implementation and test evidence. Final team review, rehearsals and submission remain in the [milestone checklist](docs/iteration-17/operations/next-milestones.md).
