/**
 * Infrastructure layer – Supabase authentication gateway.
 *
 * Every direct Supabase auth / profile call lives here so that
 * Presentation-layer components never touch the SDK directly.
 */

import { createClient } from "../../lib/supabase/client";

export type UserRole = "coach" | "analyst";

export interface SignInResult {
  userId: string;
  role: UserRole;
}

export async function signIn(
  email: string,
  password: string,
): Promise<SignInResult> {
  const supabase = createClient();

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error || !data.user) {
    throw new Error(error?.message ?? "Unable to log in right now.");
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", data.user.id)
    .single();

  const metadataRole =
    data.user.user_metadata?.role === "coach" ||
    data.user.user_metadata?.role === "analyst"
      ? (data.user.user_metadata.role as UserRole)
      : null;

  const role: UserRole =
    (profile?.role as UserRole | undefined) ?? metadataRole ?? "analyst";

  return { userId: data.user.id, role };
}

export async function signUp(
  email: string,
  password: string,
  role: UserRole,
): Promise<void> {
  const supabase = createClient();

  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { role },
      emailRedirectTo: `${window.location.origin}/auth/callback?next=/`,
    },
  });

  if (error) {
    throw new Error(error.message);
  }
}
