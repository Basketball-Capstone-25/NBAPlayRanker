-- SCRUM-771: administrator-run check for the saved_queries table, usable in psql or the Supabase SQL editor.
-- Fixtures exist only inside this transaction, which is rolled back at the end.
begin;
create temporary table saved_query_test_ids (label text primary key, id uuid default gen_random_uuid());
insert into saved_query_test_ids (label) values ('analyst_a'), ('analyst_b'), ('coach');
grant select on saved_query_test_ids to authenticated, anon;
insert into auth.users (id, email, raw_user_meta_data)
select id, id::text || '@saved-query-test.invalid', '{"role":"analyst"}'::jsonb from saved_query_test_ids;
update public.profiles set role = 'analyst'
where id in (select id from saved_query_test_ids where label like 'analyst_%');
update public.profiles set role = 'coach'
where id = (select id from saved_query_test_ids where label = 'coach');

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
create function pg_temp.act_as(who text) returns void
language plpgsql as $$ begin
  perform set_config('request.jwt.claims',
    json_build_object('sub', (select id from saved_query_test_ids t where t.label = who), 'role', 'authenticated')::text, true);
end $$;

-- Analyst A saves a query.
select pg_temp.act_as('analyst_a');
set local role authenticated;
insert into public.saved_queries (name, dataset, filters, sort_key, sort_dir, row_limit)
values ('Our 5 best play types by PPP this season', 'synergy', '{"season":"2024-25","team":"TOR","side":"offense"}', 'PPP', 'desc', 5);
select pg_temp.assert_true((select count(*) from public.saved_queries) = 1, 'analyst sees their own query');
select pg_temp.expect_error($q$insert into public.saved_queries (name, dataset, row_limit)
  values ('  our 5 BEST play types by ppp   this season ', 'synergy', 5)$q$, '23505');
select pg_temp.expect_error($q$insert into public.saved_queries (name, dataset, row_limit) values ('   ', 'synergy', 5)$q$, '23514');
select pg_temp.expect_error($q$insert into public.saved_queries (name, dataset, row_limit) values ('Bad dataset', 'boxscore', 5)$q$, '23514');
select pg_temp.expect_error($q$insert into public.saved_queries (name, dataset, row_limit) values ('Too many rows', 'shots', 10001)$q$, '23514');
update public.saved_queries set name = 'Top 5 play types by PPP, 2024-25';
select pg_temp.assert_true((select name from public.saved_queries) = 'Top 5 play types by PPP, 2024-25', 'analyst can rename');
select pg_temp.expect_error($q$update public.saved_queries set filters = '{}'$q$, '42501');
reset role;

-- Analyst B and the coach cannot see or change it.
select pg_temp.act_as('analyst_b');
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.saved_queries) = 0, 'another analyst sees nothing');
update public.saved_queries set name = 'Taken over';
delete from public.saved_queries;
reset role;
select pg_temp.assert_true((select name from public.saved_queries) = 'Top 5 play types by PPP, 2024-25', 'other analyst changed nothing');

select pg_temp.act_as('coach');
set local role authenticated;
select pg_temp.assert_true((select count(*) from public.saved_queries) = 0, 'coach sees nothing');
select pg_temp.expect_error($q$insert into public.saved_queries (name, dataset, row_limit) values ('Coach query', 'synergy', 5)$q$, '42501');
reset role;

-- Analyst A deletes it.
select pg_temp.act_as('analyst_a');
set local role authenticated;
delete from public.saved_queries;
select pg_temp.assert_true((select count(*) from public.saved_queries) = 0, 'analyst can delete');
reset role;

select 'saved_queries verification passed' as result;
rollback;
