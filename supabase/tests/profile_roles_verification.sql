-- Run as the database administrator after the migration. Every test user and
-- role change is rolled back. Any failed assertion aborts this transaction.
begin;

create temporary table role_test_ids (subject uuid, other_user uuid);
insert into role_test_ids values (gen_random_uuid(), gen_random_uuid());
grant select on role_test_ids to authenticated;

insert into auth.users (id, email, raw_user_meta_data)
select subject, subject::text || '@role-test.invalid', '{"role":"coach","requested_role":"coach"}'::jsonb
from role_test_ids
union all
select other_user, other_user::text || '@role-test.invalid', '{"role":"analyst"}'::jsonb
from role_test_ids;

do $$
begin
  if (select count(*) from public.profiles p join role_test_ids t
      on p.id in (t.subject, t.other_user) where p.role is null) <> 2 then
    raise exception 'FAIL: signup metadata granted a role';
  end if;
end;
$$;

-- Simulate an administrator assigning a role, which must continue to work.
update public.profiles set role = 'analyst'
where id = (select subject from role_test_ids);
update public.profiles set role = 'coach'
where id = (select other_user from role_test_ids);

select set_config('request.jwt.claims', json_build_object(
  'sub', (select subject from role_test_ids),
  'role', 'authenticated',
  'user_metadata', json_build_object('role', 'coach')
)::text, true);
set local role authenticated;

do $$
begin
  if (select count(*) from public.profiles) <> 1 then
    raise exception 'FAIL: users can read another profile or cannot read their own';
  end if;
  if (select role::text from public.profiles) <> 'analyst' then
    raise exception 'FAIL: trusted profile assignment was not preserved';
  end if;

  begin
    update public.profiles set role = 'coach' where id = (select subject from role_test_ids);
    raise exception 'FAIL: authenticated user changed their role';
  exception when insufficient_privilege then
    null;
  end;

  begin
    insert into public.profiles (id, email, role)
    values (gen_random_uuid(), 'forged@role-test.invalid', 'coach');
    raise exception 'FAIL: authenticated user inserted a role';
  exception when insufficient_privilege then
    null;
  end;

  begin
    delete from public.profiles where id = (select subject from role_test_ids);
    raise exception 'FAIL: authenticated user deleted their authorization record';
  exception when insufficient_privilege then
    null;
  end;
end;
$$;

reset role;
set local role anon;
do $$
begin
  begin
    perform 1 from public.profiles;
    raise exception 'FAIL: anonymous user read profiles';
  exception when insufficient_privilege then
    null;
  end;
end;
$$;

reset role;
rollback;
select 'PASS: forged signup metadata, own-profile role update, insert, delete, other-profile reads, and anonymous reads rejected; administrator assignment preserved; all test data rolled back' as result;
