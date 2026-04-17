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

function normalizeUserRole(value: unknown): UserRole | null {
  return value === "coach" || value === "analyst" ? value : null;
}

function getEmailRedirectUrl(): string {
  if (typeof window !== "undefined") {
    return new URL("/auth/callback?next=/login", window.location.origin).toString();
  }

  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL?.trim();
  if (siteUrl) {
    return new URL("/auth/callback?next=/login", siteUrl.replace(/\/+$/, "")).toString();
  }

  throw new Error(
    "Unable to determine auth redirect URL. Set NEXT_PUBLIC_SITE_URL or call signUp from the browser.",
  );
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

  const { data: profile, error: profileError } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", data.user.id)
    .maybeSingle();

  if (profileError) {
    throw new Error(profileError.message);
  }

  const profileRole = normalizeUserRole(profile?.role);
  const metadataRole = normalizeUserRole(data.user.user_metadata?.role);

  const role: UserRole = profileRole ?? metadataRole ?? "analyst";

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
      emailRedirectTo: getEmailRedirectUrl(),
    },
  });

  if (error) {
    throw new Error(error.message);
  }
}
