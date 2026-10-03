-- Minimal isolated PostgreSQL fixture, NOT a production migration.
-- Run only in a newly created disposable test database.
create schema auth;
do $$ begin
  if not exists (select from pg_roles where rolname='anon') then create role anon nologin; end if;
  if not exists (select from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
  if not exists (select from pg_roles where rolname='service_role') then create role service_role nologin; end if;
end $$;
grant usage on schema auth, public to authenticated, anon, service_role;
create function auth.uid() returns uuid language sql stable as $$
  select (current_setting('request.jwt.claims',true)::jsonb->>'sub')::uuid;
$$;
create table auth.users (id uuid primary key, email text, raw_user_meta_data jsonb);
create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text, role text, created_at timestamptz default now()
);
-- Apply protect_profile_roles.sql, then gameplan_state_schema.sql, then tests.
