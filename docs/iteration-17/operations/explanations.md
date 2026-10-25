# Checks for generated gameplan explanations

The proposed feature explains an existing ranking. It does not choose a different play, calculate statistics, promise an outcome or turn retrospective calibration into a forecast. The current templates remain the fallback. This is the design for BSR-158 / SCRUM-537; no model API has been enabled or paid evaluation run.

## Input and output

Send only the current validated matchup, period/clock/score, supported coaching conditions, ranked play-type IDs, evidence IDs, sample sizes, dataset version and model version. Do not send account information, access tokens or private Gameplan notes. Reject an empty, stale or unavailable ranking before calling a provider.

Every evidence item has an ID, a source field, a value and a caveat. Keep raw data and user text separate from instructions. Treat any instructions inside a team name, note or returned model text as data.

Require a short structured response with:

- `play_id`: one of the displayed ranked play-type IDs; the main recommendation must be the displayed first-ranked play.
- `evidence_ids`: one to three IDs from the supplied evidence.
- `explanation`: at most two short sentences explaining those evidence items.
- `caveat_ids`: the supplied limits relevant to the recommendation.

Validate the schema and reject unknown IDs, changed rankings, unsupported claims or conflicting context. Render all numbers, labels and caveats from the server's evidence record. Never use model-generated text as the source of a displayed statistic. If any check fails, show the existing template and record a reason code without saving private input.

## Evaluation cases

| Case | Required result |
| --- | --- |
| Normal TOR/BOS matchup | Explains the displayed first-ranked play using supplied evidence only. |
| Small sample | Retains the sample-size caveat and avoids strong certainty. |
| Baseline result with score/time present | Says those fields were not used when the baseline did not use them. |
| Late-game three-point request | Describes only a condition actually applied by the ranking service. |
| Overtime | Preserves overtime and the supplied clock. |
| Empty or blocked shot data | Uses an unavailable-state explanation; no provider call or invented ranking. |
| Old response after filters change | Discards the stale response and waits for the current ranking. |
| Invented play or evidence ID | Rejects the response and uses the template. |
| Changed statistic, certainty or rank | Rejects the response, even if its JSON is valid. |
| Instructions hidden in input | Does not follow them or reveal configuration. |
| Repeated identical request | Uses the cached accepted explanation for unchanged evidence. |
| Unclear request | Requests clarification or uses only validated fields; does not guess a team, score or time. |
| Provider timeout, invalid JSON or rate limit | Shows the template without breaking the Gameplan page. |
| Spending limit reached | Makes no paid request and continues with the template. |

For each selected provider/model version, run every case three times. Record the evidence snapshot, validator decision, factual errors, response time and usage. No changed play IDs, invented numbers or missing required caveats are acceptable. A coach or peer should then check whether each accepted explanation is understandable in one reading. Record their feedback rather than claiming that an automated schema check proves clarity.

## Timeout, cache and cost rules

Start with the provider disabled and a model-API budget of **CAD $0**. The existing CAD $10 Google Cloud approval does not authorize a separate model API.

Before activation, record the chosen provider, dated input/output rates, currency conversion allowance and a separately approved monthly amount. Calculate the maximum request cost from the token limits. Reserve that amount before making a call and reconcile it with returned usage. Deny a call if its reservation would exceed the monthly balance. This application limit supplements provider controls; delayed provider billing can still differ from estimates.

Proposed initial limits for review: 1,500 input tokens, 200 output tokens, five-second timeout, one provider request at a time, ten requests per user per day and twenty per project per day. No automatic paid retry. Cache for one hour using the user/role, canonical context, ranked evidence, dataset version, model version and prompt version. Expire the cache when any of those change. A repeated request should not spend again while its accepted cached result is current.

Store only model/version, reason codes, timing, token counts and cost totals in operational logs. Expose a server-side disable switch. At timeout, bad output, quota exhaustion or provider outage, return the existing template. Check these paths with a fake provider before spending on evaluation.

The design is ready to review. Provider selection, the spending amount and the reviewer clarity check remain future decisions. This work does not claim that generated explanations are implemented or evaluated.
