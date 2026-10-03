# Gameplan storage contract

SCRUM-481 provides the database contract for UC42/UC48. It does not migrate the
Gameplan frontend away from localStorage; that integration belongs to SCRUM-482.

The schema is defined in
`supabase/migrations/20261003211500_gameplan_state_schema.sql`. It relies on the
protected role assignment introduced by
`supabase/migrations/20261002201325_protect_profile_roles.sql`.

## One private state per matchup

`public.gameplan_notes` has the composite primary key
`(user_id, season, our_team, opp_team)`. Matchups are directed: TOR versus BOS and
BOS versus TOR are separate states, as are different seasons and owners.

| Column | Type and rules |
| --- | --- |
| `user_id` | UUID; defaults to the signed-in user's `auth.uid()`; references `auth.users(id)` with cascading deletion. |
| `season` | Text matching `YYYY-YY`, for example `2024-25`. |
| `our_team`, `opp_team` | Two to four uppercase letters; teams must differ. These constraints check format, not membership in the available dataset. |
| `notes` | JSON object mapping play IDs to strings; default `{}`. At most 100 entries, IDs 1–160 characters, text at most 4,000 characters per entry. |
| `plan` | Ordered JSON array, default `[]`; at most 50 items. Each item contains exactly `id` and `label` strings. IDs are unique and 1–160 characters; labels are 1–240 characters. |
| `roles` | JSON object with exactly `ballHandler`, `screener`, `cornerSpacer`, `cutter`, and `safety`. Each value is a string of at most 120 characters; all default to empty strings. |
| `revision` | Positive bigint, default `1`; the server increments it on every accepted update. |
| `created_at`, `updated_at` | Server-managed timestamps with time zone. The update trigger refreshes `updated_at`. |

Every column is non-null. The combined serialized `notes`, `plan`, and `roles`
values are limited to 65,536 bytes. `private.valid_gameplan_state` enforces shape
and size; `private.touch_gameplan_state` maintains update metadata.

## Authorization and write behavior

Row-level security allows a signed-in user to read, insert, update, or delete only
their own state while their current protected `public.profiles.role` is `coach`.
Analysts, pending users, anonymous callers, and other coaches cannot access it.
Signup/JWT user metadata cannot grant access. A role demotion takes effect on the
next database operation, even when an older session token is still present.

Authenticated clients can insert identity columns and state fields, but can update
only `notes`, `plan`, and `roles`. They cannot change ownership, matchup identity,
revision, or timestamps. Use the publishable key and the signed-in user's session;
never expose a service-role key to the browser.

Use **INSERT for a new state** and **PATCH/UPDATE for an existing state**. Do not
use a generic `.upsert()` that attempts to update immutable identity columns.

```ts
// Existing Supabase browser client, after checking the current signed-in user.
const identity = { season, our_team: ourTeam, opp_team: oppTeam };
const state = { notes, plan, roles };

// New state: user_id defaults to auth.uid(); metadata defaults on the server.
const created = await supabase.from("gameplan_notes")
  .insert({ ...identity, ...state })
  .select("notes,plan,roles,revision,updated_at")
  .single();

// Existing state: update only mutable fields and require the revision read earlier.
const updated = await supabase.from("gameplan_notes")
  .update(state)
  .eq("user_id", user.id)
  .eq("season", season)
  .eq("our_team", ourTeam)
  .eq("opp_team", oppTeam)
  .eq("revision", lastReadRevision)
  .select("notes,plan,roles,revision,updated_at")
  .maybeSingle();
```

Check `error` for both operations. A duplicate-key INSERT (`23505`) means another
tab or session may have created the state: reload it before retrying. An UPDATE
with no returned row can mean a revision conflict, deletion, or lost access;
reload and handle the outcome rather than reporting a successful save. Preserve
the user's draft while resolving conflicts. Store the newly returned revision
after a successful write. The revision predicate is a client protocol: the table
does not force a caller to include it in every UPDATE.

## Verification

The migration was applied to the capstone Supabase project on October 3, 2026.
`supabase/tests/gameplan_state_verification.sql` passed against both an isolated
local PostgreSQL database and the live project. The test transaction rolls back
all fixture users, profile assignments, state rows, and helper functions.

To reproduce locally, start a disposable PostgreSQL instance and run these from
the repository root with an administrator connection to that instance. The
bootstrap is only for a newly created test database; it is not a production
migration.

```sh
createdb capstone_gameplan_test
psql -X -v ON_ERROR_STOP=1 -d capstone_gameplan_test -f supabase/tests/local_gameplan_bootstrap.sql
psql -X -v ON_ERROR_STOP=1 -d capstone_gameplan_test -f supabase/migrations/20261002201325_protect_profile_roles.sql
psql -X -v ON_ERROR_STOP=1 -d capstone_gameplan_test -f supabase/migrations/20261003211500_gameplan_state_schema.sql
psql -X -v ON_ERROR_STOP=1 -d capstone_gameplan_test -f supabase/tests/gameplan_state_verification.sql
```

Expected result: `PASS: Gameplan owner CRUD, isolation, trusted roles, constraints,
optimistic revision, audit columns and cascade; all fixtures rolled back`.
The test covers cross-user isolation, role demotion, forged metadata, immutable
identity/audit columns, malformed JSON, duplicate plan IDs, and stale revisions.
It does not prove a working frontend save flow.

## SCRUM-482 handoff

The current UI stores `notesByPlay`, the ordered `plan`, and player `roles` in
`app/ui/gameplan/_components/GameplanClient.tsx`. Adapt those fields to this schema
through the existing infrastructure/service layering. Load state using the current
user plus season and directed matchup; cancel stale loads when these change.
Offer an explicit save or debounced persistence with visible saving, saved,
failure, and conflict states. Avoid overwriting a new matchup with the previous
matchup's draft.

Existing localStorage data has no reliable owner/matchup scope. Do not silently
upload it to whichever account is signed in: offer a reviewed import into a chosen
matchup. Preserve local drafts until a server save succeeds, clear account-bound
state on sign-out, and verify cross-device restore and two-tab revision conflicts.
Keep play IDs stable enough for notes to remain associated with the intended play.
