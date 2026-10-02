-- This SECURITY DEFINER function serves the existing ensure_rls DDL event
-- trigger. Client roles do not need direct execution permission. Preserve the
-- function, trigger, postgres ownership, and service_role's existing grant.
revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
