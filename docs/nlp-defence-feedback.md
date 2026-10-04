# NLP defence committee feedback (SCRUM-502)

## 1. Error handling and fallbacks
A prompt the spaCy/NLTK pipeline cannot parse (for example "Just win the game")
no longer leaves the recommender without inputs. `resolve_context_ml_params`
in `backend/infrastructure/external_integrations/nlp_parser.py` fills each
missing required field with a neutral default and reports it:

| Field | Default | Meaning |
|---|---|---|
| `margin` | `0.0` | tied game |
| `period` | `1` | first quarter |
| `time_remaining` | `720.0` | full quarter left |

UI defaults sent with the request still take priority. `POST /nlp/parse`
returns `defaulted_fields` so the page can tell the coach which values were
assumed, plus the existing clarifying questions and lower confidence score.

## 2. Extraction accuracy
`backend/tests/test_nlp_parser.py` checks the mapping programmatically
("down 5" -> -5, "leading by 3" -> 3, "tied" -> 0, period and clock phrases).
Before values reach the ranking logic they are range-checked
(period 1-5, margin -60..60, time remaining 0-720 s, shot clock 0-24 s).
Out-of-range or non-finite values are replaced by the neutral default and
listed in `rejected_fields`. The suite runs in CI on every push.

## 3. Performance and environment
The spaCy pipeline is built once per process (`get_nlp_pipeline` singleton) and
excludes the NER component; if `en_core_web_sm` is unavailable it falls back to
`spacy.blank("en")`. The backend is deployed on Cloud Run with 2 GiB memory
and at most two instances.

`backend/tests/test_nlp_load.py` simulates 8 concurrent coaches sending 400
`/nlp/parse` and `/nlp/explain` requests to the full FastAPI app in one process
and records process memory (results in `docs/nlp-load-test-results.json`).
Run on 2026-10-03 on a Windows 11 development machine, Python 3.14:

| Measure | Result |
|---|---|
| Failed requests | 0 of 400 |
| Throughput | 138 requests/s |
| Latency p50 / p95 / max | 61 ms / 77 ms / 84 ms |
| Process memory before first NLP request | 306 MB |
| Peak process memory under load | 308 MB (limit 2048 MB) |
| Memory growth during the run | about 1 MB |

Limits of this result: on that machine `en_core_web_sm` was not installed, so
the pipeline ran on its `spacy.blank("en")` fallback and the figures do not
include the trained model that the Cloud Run image downloads. Memory on the
deployed service itself has not been measured.

## 4. Integration boundary
The rationale text is produced by deterministic templates in `nlp_explain.py`,
not by a language model. Every number in an explanation is read from the
ranking payload, so the text cannot introduce values the recommender did not
compute. `test_explanations_are_deterministic` and
`test_explanation_evidence_contains_real_metrics` cover this.
