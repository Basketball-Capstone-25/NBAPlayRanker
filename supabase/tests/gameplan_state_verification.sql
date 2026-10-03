-- Administrator-run integration test, usable with psql or Supabase SQL editor.
-- Fixtures, profile assignments and test helpers exist only in this transaction.
begin;
create temporary table gameplan_test_ids (label text primary key, id uuid default gen_random_uuid());
insert into gameplan_test_ids (label) values ('coach_a'), ('coach_b'), ('analyst'), ('pending');
grant select on gameplan_test_ids to authenticated, anon;
insert into auth.users (id, email, raw_user_meta_data)
select id, id::text || '@gameplan-test.invalid', '{"role":"coach"}'::jsonb from gameplan_test_ids;
update public.profiles set role = 'coach'
where id in (select id from gameplan_test_ids where label like 'coach_%');
update public.profiles set role = 'analyst'
where id = (select id from gameplan_test_ids where label = 'analyst');

create function pg_temp.assert_true(ok boolean, message text) returns void
language plpgsql as $$ begin
  if ok is distinct from true then raise exception 'FAIL: %', message; end if;
end $$;
create function pg_temp.expect_error(statement text, expected_state text) returns void
language plpgsql as $$ begin
  begin
    execute statement;
  exception when others then
    if sqlstate = expected_state then return; end if;
    raise exception 'Wrong SQLSTATE %, expected %: %', sqlstate, expected_state, sqlerrm;
  end;
  raise exception 'FAIL: statement was unexpectedly accepted: %', statement;
end $$;

select pg_temp.assert_true((select count(*) from public.profiles p join gameplan_test_ids t on p.id=t.id where t.label='pending' and p.role is null)=1,
  'forged signup metadata must not grant a role');
select set_config('request.jwt.claims', json_build_object('sub', (select id from gameplan_test_ids where label='coach_b'), 'role','authenticated')::text, true);
set local role authenticated;
insert into public.gameplan_notes (season,our_team,opp_team,notes)
values ('2024-25','TOR','BOS','{"Isolation":"Other coach private note"}');
reset role;

select set_config('request.jwt.claims', json_build_object('sub', (select id from gameplan_test_ids where label='coach_a'), 'role','authenticated')::text, true);
set local role authenticated;
insert into public.gameplan_notes (season,our_team,opp_team,notes,plan,roles)
values ('2024-25','TOR','BOS','{"Isolation":"Attack the switch"}',
  '[{"id":"Isolation__0__smart","label":"Isolation"},{"id":"Cut__1__baseline","label":"Cut"}]',
  '{"ballHandler":"Player 1","screener":"Player 2","cornerSpacer":"","cutter":"","safety":""}');
select pg_temp.assert_true((select count(*) from public.gameplan_notes)=1, 'read own row only');
select pg_temp.assert_true((select plan->1->>'label' from public.gameplan_notes)='Cut', 'preserve plan order');
select pg_temp.assert_true((select revision from public.gameplan_notes)=1, 'initial revision');
select pg_temp.assert_true((select roles->>'ballHandler' from public.gameplan_notes)='Player 1', 'preserve role assignment');

update public.gameplan_notes set notes='{"Isolation":"Updated"}' where revision=1;
select pg_temp.assert_true((select revision from public.gameplan_notes)=2, 'server increments revision');
select pg_temp.assert_true((select updated_at >= created_at from public.gameplan_notes), 'server update timestamp');
with changed as (update public.gameplan_notes set notes='{}' where revision=1 returning 1)
select pg_temp.assert_true((select count(*) from changed)=0, 'stale compare-and-swap does not overwrite');

-- Separate seasons and reversed matchups do not collide.
insert into public.gameplan_notes (season,our_team,opp_team) values ('2023-24','TOR','BOS'),('2024-25','BOS','TOR');
select pg_temp.assert_true((select count(*) from public.gameplan_notes)=3, 'season/direction isolation');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','TOR','BOS')$q$, '23505');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','TOR','TOR')$q$, '23514');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('bad-season','TOR','BOS')$q$, '23514');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','tor','LAL')$q$, '23514');

-- Ownership and history cannot be forged through client column privileges.
select pg_temp.expect_error($q$insert into public.gameplan_notes (user_id,season,our_team,opp_team) select id,'2022-23','TOR','BOS' from gameplan_test_ids where label='coach_b'$q$, '42501');
select pg_temp.expect_error($q$update public.gameplan_notes set user_id=(select id from gameplan_test_ids where label='coach_b')$q$, '42501');
select pg_temp.expect_error($q$update public.gameplan_notes set revision=99$q$, '42501');
select pg_temp.expect_error($q$update public.gameplan_notes set created_at='2000-01-01'$q$, '42501');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team,revision) values ('2022-23','TOR','BOS',99)$q$, '42501');

-- Malformed/bloated JSON is rejected, including individual element types.
select pg_temp.expect_error($q$update public.gameplan_notes set notes='[]'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set notes='{"a":42}'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set notes=jsonb_build_object('a',repeat('x',4001))$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set plan='["invalid"]'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set plan='[{"id":"a"}]'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set plan='[{"id":"a","label":"A"},{"id":"a","label":"Duplicate"}]'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set roles='{}'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set roles=roles || '{"ballHandler":null}'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set roles=roles || '{"unexpected":"value"}'$q$, '23514');
select pg_temp.expect_error($q$update public.gameplan_notes set notes=null$q$, '23502');

with changed as (update public.gameplan_notes set notes='{}' where user_id=(select id from gameplan_test_ids where label='coach_b') returning 1)
select pg_temp.assert_true((select count(*) from changed)=0, 'other coach update denied');
with removed as (delete from public.gameplan_notes where user_id=(select id from gameplan_test_ids where label='coach_b') returning 1)
select pg_temp.assert_true((select count(*) from removed)=0, 'other coach delete denied');
with removed as (delete from public.gameplan_notes where season='2023-24' returning 1)
select pg_temp.assert_true((select count(*) from removed)=1, 'own delete succeeds');
reset role;

-- Current trusted role wins over claimed metadata. Demotion revokes access immediately.
update public.profiles set role='analyst' where id=(select id from gameplan_test_ids where label='coach_a');
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.gameplan_notes)=0, 'role demotion removes read access');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2022-23','TOR','BOS')$q$, '42501');
reset role;
select set_config('request.jwt.claims', json_build_object('sub', (select id from gameplan_test_ids where label='analyst'), 'role','authenticated','user_metadata',json_build_object('role','coach'))::text, true);
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.gameplan_notes)=0, 'analyst cannot read coach state');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','TOR','BOS')$q$, '42501');
reset role;
select set_config('request.jwt.claims', json_build_object('sub', (select id from gameplan_test_ids where label='pending'), 'role','authenticated')::text, true);
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.gameplan_notes)=0, 'pending user cannot read');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','TOR','BOS')$q$, '42501');
reset role;
set local role anon;
select pg_temp.expect_error($q$select * from public.gameplan_notes$q$, '42501');
select pg_temp.expect_error($q$insert into public.gameplan_notes (season,our_team,opp_team) values ('2024-25','TOR','BOS')$q$, '42501');
reset role;

delete from auth.users where id=(select id from gameplan_test_ids where label='coach_b');
select pg_temp.assert_true((select count(*) from public.gameplan_notes where user_id=(select id from gameplan_test_ids where label='coach_b'))=0, 'user deletion cascades to state');
rollback;
select 'PASS: Gameplan owner CRUD, isolation, trusted roles, constraints, optimistic revision, audit columns and cascade; all fixtures rolled back' as result;
