-- Keep existing coach/analyst assignments. New accounts remain pending until a
-- trusted project administrator assigns profiles.role through the SQL console
-- or a server-side service-role operation. Never copy user_metadata into roles.
alter table public.profiles alter column role drop not null;
alter table public.profiles enable row level security;

-- Profiles are authorization records, not user-editable profile settings.
-- Revoke both table and any explicit column grants: either can permit a write.
revoke all on table public.profiles from public, anon, authenticated;
revoke all (id, email, role, created_at) on public.profiles from public, anon, authenticated;
grant select on table public.profiles to authenticated;

drop policy if exists "Users can update their own profile" on public.profiles;
drop policy if exists "Users can view their own profile" on public.profiles;
create policy "Users can view their own profile"
  on public.profiles for select to authenticated
  using ((select auth.uid()) = id);

create schema if not exists private;

-- SECURITY DEFINER is required only for the Auth INSERT trigger, because
-- clients and the Auth service cannot insert directly into protected profiles.
-- A trigger has no logged-in auth.uid() during signup; validate its source
-- instead. The function is private, has a fixed search path, and is not callable
-- by client roles. No supplied metadata grants authorization.
create or replace function private.handle_new_user_profile()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if tg_op <> 'INSERT' or tg_table_schema <> 'auth' or tg_table_name <> 'users' then
    raise exception 'This function only handles new Auth users';
  end if;
  insert into public.profiles (id, email, role)
  values (new.id, new.email, null);
  return new;
end;
$$;

revoke all on function private.handle_new_user_profile() from public, anon, authenticated;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function private.handle_new_user_profile();

-- Remove the replaced public trigger function so there is no stale signup path.
drop function if exists public.handle_new_user();
