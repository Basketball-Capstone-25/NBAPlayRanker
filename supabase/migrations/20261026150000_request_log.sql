-- SCRUM-494 / UC17: endpoint activity log for auditability and troubleshooting.
-- The API writes one row per authenticated request with that user's own token,
-- so no service-role key is needed. Only analysts can read the log.
create table public.request_log (
  id bigint generated always as identity primary key,
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  user_email text check (user_email is null or length(user_email) <= 320),
  user_role text not null check (user_role in ('coach', 'analyst')),
  method text not null check (method in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')),
  endpoint text not null check (length(endpoint) between 1 and 200 and endpoint like '/%'),
  query text not null default '' check (length(query) <= 500),
  status_code smallint not null check (status_code between 100 and 599),
  duration_ms integer not null check (duration_ms >= 0),
  created_at timestamptz not null default now()
);

comment on table public.request_log is
  'SCRUM-494: who called which API endpoint and when. Rows are written by the API with the caller''s token; analysts read them.';
comment on column public.request_log.query is
  'Query string without credentials, truncated to 500 characters.';

create index request_log_created_at_idx on public.request_log (created_at desc, id desc);
create index request_log_user_created_idx on public.request_log (user_id, created_at desc);
create index request_log_endpoint_created_idx on public.request_log (endpoint, created_at desc);

alter table public.request_log enable row level security;
revoke all on public.request_log from public, anon, authenticated;
grant select on public.request_log to authenticated;
grant insert (user_email, user_role, method, endpoint, query, status_code, duration_ms)
  on public.request_log to authenticated;
grant all on public.request_log to service_role;

-- A caller can only add rows about themselves, and only with their real role.
create policy "Users log their own requests"
  on public.request_log for insert to authenticated
  with check (
    user_id = (select auth.uid()) and
    user_role = (select role::text from public.profiles where id = (select auth.uid()))
  );

-- profiles.role is administrator-managed; the lookup is limited to the caller's own profile.
create policy "Analysts read the request log"
  on public.request_log for select to authenticated
  using ((select role::text from public.profiles where id = (select auth.uid())) = 'analyst');
