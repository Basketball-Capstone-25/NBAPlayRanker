-- SCRUM-481 / UC42, UC48: storage contract for SCRUM-482's later UI integration.
-- One private state per coach, season and directed matchup; no existing data changes.
create schema if not exists private;

create function private.valid_gameplan_state(notes jsonb, plan jsonb, roles jsonb)
returns boolean
language plpgsql immutable strict security invoker
set search_path = ''
as $$
begin
  if jsonb_typeof(notes) <> 'object' or jsonb_typeof(plan) <> 'array'
     or jsonb_typeof(roles) <> 'object' then
    return false;
  end if;
  if octet_length(notes::text) + octet_length(plan::text) + octet_length(roles::text) > 65536
     or jsonb_array_length(plan) > 50
     or (select count(*) from jsonb_each(notes)) > 100 then
    return false;
  end if;
  if exists (select 1 from jsonb_each(notes) n
             where length(n.key) not between 1 and 160
                or jsonb_typeof(n.value) <> 'string'
                or length(n.value #>> '{}') > 4000) then
    return false;
  end if;
  if exists (select 1 from jsonb_array_elements(plan) p
             where jsonb_typeof(p) <> 'object'
                or not (p ?& array['id', 'label'])
                or p - array['id', 'label'] <> '{}'::jsonb
                or jsonb_typeof(p->'id') <> 'string'
                or jsonb_typeof(p->'label') <> 'string'
                or length(p->>'id') not between 1 and 160
                or length(p->>'label') not between 1 and 240) then
    return false;
  end if;
  if (select count(*) from jsonb_array_elements(plan)) <>
     (select count(distinct p->>'id') from jsonb_array_elements(plan) p) then
    return false;
  end if;
  if not (roles ?& array['ballHandler','screener','cornerSpacer','cutter','safety'])
     or roles - array['ballHandler','screener','cornerSpacer','cutter','safety'] <> '{}'::jsonb
     or exists (select 1 from jsonb_each(roles) r
                where jsonb_typeof(r.value) <> 'string'
                   or length(r.value #>> '{}') > 120) then
    return false;
  end if;
  return true;
end;
$$;

-- Constraints execute as the caller; expose only this pure validator to clients.
revoke all on function private.valid_gameplan_state(jsonb, jsonb, jsonb) from public, anon;
grant usage on schema private to authenticated;
grant execute on function private.valid_gameplan_state(jsonb, jsonb, jsonb) to authenticated, service_role;

create table public.gameplan_notes (
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  season text not null check (season ~ '^[0-9]{4}-[0-9]{2}$'),
  our_team text not null check (our_team ~ '^[A-Z]{2,4}$'),
  opp_team text not null check (opp_team ~ '^[A-Z]{2,4}$'),
  notes jsonb not null default '{}'::jsonb,
  plan jsonb not null default '[]'::jsonb,
  roles jsonb not null default '{"ballHandler":"","screener":"","cornerSpacer":"","cutter":"","safety":""}'::jsonb,
  revision bigint not null default 1 check (revision > 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, season, our_team, opp_team),
  constraint gameplan_different_teams check (our_team <> opp_team),
  constraint gameplan_state_shape check (private.valid_gameplan_state(notes, plan, roles))
);

comment on table public.gameplan_notes is
  'SCRUM-481: coach-owned Gameplan state keyed by season and directed matchup. UI integration is SCRUM-482.';
comment on column public.gameplan_notes.revision is
  'Server incremented. Future clients should UPDATE WHERE revision = last_read_revision and detect zero rows as a conflict.';

create function private.touch_gameplan_state()
returns trigger
language plpgsql security invoker
set search_path = ''
as $$
begin
  new.revision := old.revision + 1;
  new.updated_at := clock_timestamp();
  return new;
end;
$$;
revoke all on function private.touch_gameplan_state() from public, anon, authenticated;
create trigger gameplan_state_updated
  before update on public.gameplan_notes
  for each row execute function private.touch_gameplan_state();

alter table public.gameplan_notes enable row level security;
revoke all on public.gameplan_notes from public, anon, authenticated;
grant select, delete on public.gameplan_notes to authenticated;
grant insert (user_id, season, our_team, opp_team, notes, plan, roles)
  on public.gameplan_notes to authenticated;
grant update (notes, plan, roles) on public.gameplan_notes to authenticated;
grant all on public.gameplan_notes to service_role;

-- profiles.role is an administrator-managed authorization record. Its own RLS
-- restricts this lookup to the requesting user's profile. JWT metadata is ignored.
create policy "Coach owns Gameplan state"
  on public.gameplan_notes for all to authenticated
  using (
    user_id = (select auth.uid()) and
    (select role::text from public.profiles where id = (select auth.uid())) = 'coach'
  )
  with check (
    user_id = (select auth.uid()) and
    (select role::text from public.profiles where id = (select auth.uid())) = 'coach'
  );
