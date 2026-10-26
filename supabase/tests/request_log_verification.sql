-- SCRUM-494: administrator-run check of request_log access rules.
-- Everything runs in one transaction and is rolled back.
begin;
create temporary table rl_ids (label text primary key, id uuid default gen_random_uuid());
insert into rl_ids (label) values ('coach'), ('analyst'), ('pending');
grant select on rl_ids to authenticated, anon;
insert into auth.users (id, email) select id, id::text || '@request-log-test.invalid' from rl_ids;
update public.profiles set role = 'coach' where id = (select id from rl_ids where label = 'coach');
update public.profiles set role = 'analyst' where id = (select id from rl_ids where label = 'analyst');

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

-- Coach: may log own request with the real role, may not read, may not forge.
select set_config('request.jwt.claims', json_build_object('sub', (select id from rl_ids where label='coach'), 'role','authenticated')::text, true);
set local role authenticated;
insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('coach', 'GET', '/rank-plays/quick-call', 200, 42);
select pg_temp.expect_error($q$insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('analyst','GET','/x',200,1)$q$, '42501');
select pg_temp.expect_error($q$insert into public.request_log (user_id, user_role, method, endpoint, status_code, duration_ms) values ((select id from rl_ids where label='analyst'),'coach','GET','/x',200,1)$q$, '42501');
select pg_temp.assert_true((select count(*) from public.request_log) = 0, 'coach cannot read the log');
select pg_temp.expect_error($q$update public.request_log set status_code = 500$q$, '42501');
select pg_temp.expect_error($q$delete from public.request_log$q$, '42501');
reset role;

-- Pending account: cannot write.
select set_config('request.jwt.claims', json_build_object('sub', (select id from rl_ids where label='pending'), 'role','authenticated')::text, true);
set local role authenticated;
select pg_temp.expect_error($q$insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('coach','GET','/x',200,1)$q$, '42501');
reset role;

-- Analyst: reads everyone's rows, cannot change them; bad values are rejected.
select set_config('request.jwt.claims', json_build_object('sub', (select id from rl_ids where label='analyst'), 'role','authenticated')::text, true);
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.request_log where endpoint = '/rank-plays/quick-call') >= 1, 'analyst reads other users rows');
insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('analyst', 'GET', '/admin/request-log', 200, 7);
select pg_temp.expect_error($q$insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('analyst','TRACE','/x',200,1)$q$, '23514');
select pg_temp.expect_error($q$insert into public.request_log (user_role, method, endpoint, status_code, duration_ms) values ('analyst','GET','no-slash',200,1)$q$, '23514');
select pg_temp.expect_error($q$delete from public.request_log$q$, '42501');
reset role;

-- Anonymous: no access.
set local role anon;
select pg_temp.expect_error($q$select count(*) from public.request_log$q$, '42501');
reset role;

delete from auth.users where id = (select id from rl_ids where label = 'coach');
select pg_temp.assert_true((select count(*) from public.request_log where user_id = (select id from rl_ids where label='coach')) = 0, 'user deletion cascades');
rollback;
select 'PASS: request_log write-own, analyst-only read, no edits, value checks and cascade; fixtures rolled back' as result;
