# Hosting cost and demo fallback

The approved Google Cloud budget is CAD $10 per month. No paid Vercel or Supabase upgrade is approved. The [live configuration check](evidence/hosting-live-check.json) found:

- Billing is enabled for `nba-playranker`.
- The monthly project budget is CAD $10, including credits, with 50%, 80% and 100% alerts.
- The Cloud Run spend cap is CAD $7, excludes credits and is `CONFIGURED`.
- The API uses 1 CPU, 2 GiB memory, zero minimum instances, two maximum instances, concurrency two and request-based CPU. All five numeric thread limits are `1`.

The [final service record](../verification/backend-service-final.json) shows revision `nba-playranker-api-00010-jp4` serving 100% of traffic. It preserves 1 CPU, 2 GiB memory, two maximum instances, concurrency two and request-based CPU. The earlier live configuration check records the budget and thread settings. Vercel's stable production address is [NBA Play Ranker](https://nbaplayranker-seven.vercel.app).

## Current bill and forecast

[Billing Reports](https://console.cloud.google.com/billing/018EC0-01D68C-D82111/reports?project=nba-playranker) was checked with the project filter set to NBA Play Ranker. October 1–25 shows **CAD $0.19 net**: $0.36 usage less $0.17 credits.

| Service | Usage | Credits | Net |
| --- | ---: | ---: | ---: |
| Artifact Registry | $0.17 | $0.00 | $0.17 |
| Cloud Storage | $0.02 | $0.00 | $0.02 |
| Cloud Run | $0.17 | −$0.17 | $0.00 |

The October 1–31 forecast is **CAD $0.19 net**, including forecast credits of $0.22. Tax was not displayed and the report listed no invoice charges. Billing reports can arrive late. These figures were checked before the new release and its test campaign, so review the report again afterward. They are not a fixed price for future months.

Budget alerts do not stop spending. A spend cap limits one eligible service using gross estimated costs, and enforcement is not instant. Builds, artifact storage and other services are outside this Cloud Run cap. These controls therefore do not guarantee a CAD $10 total. See Google's [budget guidance](https://docs.cloud.google.com/billing/docs/how-to/budgets) and [spend-cap limitations](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps).

## If costs rise or hosting stops

At the 50% alert, inspect which service is growing. At 80%, use the local demo for repeated tests and stop unnecessary builds or repeated training. If the forecast exceeds CAD $10, keep the limits and use the local fallback while deciding which workload to reduce. Do not lift the cap or enable a paid plan without a new spending decision.

The [local recovery guide](recovery.md) provides a working fallback. It passed real coach sign-in and returned five rankings without calling Cloud Run. It still uses Supabase for authentication, so it needs internet access and a healthy Supabase project. If Supabase is also down, use the saved evidence to explain the last verified release; do not present cached results as a live successful request.

Keep a known-good source commit, dependency record, dataset checksums and native VP copy before any migration. Keep private database backups separately. Test any replacement with the same login, role checks, recommendation and export cases before switching the stable address. Roll back if those checks fail.

BSR-160 has the cost, limits and fallback evidence. The [final API checks](../verification/live-api-final/results.json) passed sign-in and coach recommendations on revision `00010-jp4`. The [coach browser walkthrough](../verification/browser/browser-checks.json) also passed. No billing controls were changed during this work.
