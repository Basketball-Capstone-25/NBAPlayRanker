# Deployment model revision

Open **Deployment Model2.0 → Deployment Diagram** in the native project. Choose the populated diagram; the project also contains an older empty diagram with the same name. The [image](deployment.jpg) shows the revised layout.

The browser, Vercel, Cloud Run and Supabase now have separate communication paths that do not cross. Each path states HTTPS and TCP port 443. Cloud Run also shows Uvicorn's internal HTTP port 8080, which matches the live service configuration.

| Connection | Purpose |
| --- | --- |
| Browser → Vercel | Pages, scripts and CSV/PDF downloads |
| Browser → Cloud Run | REST/JSON requests with a Bearer token |
| Browser → Supabase | Sign-in and session refresh |
| Vercel → Supabase | User authentication and profile lookup |
| Cloud Run → Supabase | Signing keys and profile lookup |
| Vercel → Cloud Run | Authenticated CSV/PDF export requests |

Artifacts are grouped inside their runtimes. The duplicate internal arrows are suppressed in this view; their model relationships are retained. Browser Gameplan storage still uses localStorage. The database schema is ready for the separate persistence work.

The [verification](verification.json) confirms that all 192 diagrams remain. Only this deployment diagram changed. All 191 other diagrams, including the 17 interaction diagrams, match shared revision 54. Original model identities and creation records are preserved.

The deployment diagram is published in shared Teamwork revision 55. The native file here is the committed pristine copy; see [publication status](teamwork-status.json) and [publication verification](publication-verification.json).

The original verification above records the prepared candidate before publication. The publication check confirms that revision 55 contains those changes and preserves the other 191 diagrams. Later connector-caption formatting entries in the open working copy are excluded from this native deliverable.

Sources: [export route](../../../app/api/exports/route.ts), [API client](../../../app/infrastructure/api-client.ts), [Dockerfile](../../../backend/Dockerfile), [Supabase profile lookup](../../../backend/infrastructure/external_integrations/supabase_profiles.py), [JWT key lookup](../../../backend/infrastructure/external_integrations/supabase_jwt.py), [live container port](cloud-run-port-check.json), and [Cloud Run's TLS/port contract](https://docs.cloud.google.com/run/docs/container-contract).
