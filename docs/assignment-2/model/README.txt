Assignment 2 model changes

The native project adds three complete use-case collaborations for SCRUM-480,
SCRUM-486 and SCRUM-526. Each has an overview, structural design and detailed
sequence: nine new diagrams in total. The 192 original diagrams are retained.

Under Interaction Model, open:

1. Run Data Quality Checks (SCRUM-480)
   The overview realizes the existing use case and shows the participating
   subsystems. The structural design and sequence show the source checks,
   dataset fingerprint, refresh and retry behavior.

2. Export KPIs / Reports (SCRUM-486)
   The overview realizes the existing analyst reporting use case. The shared
   export design and sequence show the displayed rows, applied filters,
   verified session, role checks, validation and CSV/JSON/PDF generation.
   Matchup and Context reuse this control for coaches. Their older use cases
   and collaboration are unchanged.

3. Evaluate Next-Season PPP (offline) (SCRUM-526)
   This new use case and collaboration describe the offline evaluation script.
   The design and sequence show prior-season inputs, training, tuning,
   final-season scoring and saved JSON/CSV evidence. It is not a live app
   prediction or pre-game forecast.

The export is a snapshot of displayed results; it does not rerun the model.
Model Metrics exports its displayed holdout table. CSV keeps applied filters
in the filename; JSON and PDF also include them inside the file. Data quality
covers the Synergy source, not shot data or prediction accuracy.

The changes are published in shared revision 56. The Desktop commit was made
by asahang@sheridancollege.ca at 2026-10-19 02:09:28 UTC, with this comment:
Add export, data-check and next-season evaluation collaborations (SCRUM-486/480/526)

The JPG previews are official Visual Paradigm exports. The original diagrams,
model identities, authors and creation times are preserved. The four diagrams
prepared earlier in this work retain their IDs and creator metadata. They are
now organized into the three collaborations. The earlier unpublished
Assignment 1 sequence edits are excluded.

See publication-verification.json for the committed revision checks and
collaboration-trace.json for the use-case, realization and diagram IDs.
The included VPP is the committed pristine snapshot with zero pending history.

Visual Paradigm normalized commit metadata, model references and presentation
defaults during publication. Every original model identity and all original
diagram-element rows are preserved. The prepared model checks are retained
separately in prepared-preservation-verification.json.

The Desktop working copy has 36 later presentation changes to deployment
caption positions, sizes and drawing order. They were left untouched and are
not included in this committed snapshot; no protocol labels or model semantics
changed. Publication was confirmed through the Desktop result and matching
native/pristine revision markers, without a new independent network query.
