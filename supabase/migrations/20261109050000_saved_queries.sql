-- SCRUM-771: Data Explorer saved queries.
-- Each analyst saves their own filters and sorting under a name that states the
-- question it answers. Rows belong to one analyst; nobody else can read or change them.

create table public.saved_queries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  name text not null check (length(btrim(name)) between 1 and 120),
  dataset text not null check (dataset in ('synergy', 'shots')),
  filters jsonb not null default '{}'::jsonb check (jsonb_typeof(filters) = 'object' and octet_length(filters::text) <= 4096),
  sort_key text check (sort_key is null or length(sort_key) between 1 and 60),
  sort_dir text check (sort_dir is null or sort_dir in ('asc', 'desc')),
  row_limit integer not null check (row_limit between 1 and 10000),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Names are unique per analyst, ignoring case and extra spaces.
create unique index saved_queries_user_name_key
  on public.saved_queries (user_id, lower(regexp_replace(btrim(name), '\s+', ' ', 'g')));

comment on table public.saved_queries is
  'SCRUM-771: analyst-owned Data Explorer queries (dataset, filters, sort and row limit) saved under a name.';

create function private.touch_saved_query()
returns trigger
language plpgsql security invoker
set search_path = ''
as $$
begin
  new.updated_at := clock_timestamp();
  return new;
end;
$$;
revoke all on function private.touch_saved_query() from public, anon, authenticated;
create trigger saved_query_updated
  before update on public.saved_queries
  for each row execute function private.touch_saved_query();

alter table public.saved_queries enable row level security;
revoke all on public.saved_queries from public, anon, authenticated;
grant select, delete on public.saved_queries to authenticated;
grant insert (user_id, name, dataset, filters, sort_key, sort_dir, row_limit)
  on public.saved_queries to authenticated;
grant update (name) on public.saved_queries to authenticated;
grant all on public.saved_queries to service_role;

-- Same role lookup as the Gameplan table: profiles.role is administrator-managed
-- and JWT metadata is ignored.
create policy "Analyst owns saved queries"
  on public.saved_queries for all to authenticated
  using (
    user_id = (select auth.uid()) and
    (select role::text from public.profiles where id = (select auth.uid())) = 'analyst'
  )
  with check (
    user_id = (select auth.uid()) and
    (select role::text from public.profiles where id = (select auth.uid())) = 'analyst'
  );
