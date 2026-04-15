import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

type UserRole = "coach" | "analyst" | null;

const signedInOnly = ["/glossary"];

const coachOnly = ["/matchup", "/context", "/gameplan"];

const analystOnly = [
  "/data-explorer",
  "/model-metrics",
  "/statistical-analysis",
  "/shot-explorer",
  "/shot-heatmap",
  "/shot-model-metrics",
  "/shot-statistical-analysis",
  "/shot-plan",
];

function matchesRoute(pathname: string, routes: string[]) {
  return routes.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

function defaultPathForRole(role: UserRole) {
  if (role === "coach") return "/matchup";
  if (role === "analyst") return "/data-explorer";
  return "/";
}

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request });

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

  // Local-dev fallback: allow app pages to load when Supabase env vars are not configured.
  if (!supabaseUrl || !supabaseKey) {
    return response;
  }

  const supabase = createServerClient(
    supabaseUrl,
    supabaseKey,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
          response = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) => {
            response.cookies.set(name, value, options);
          });
        },
      },
    }
  );

  const path = request.nextUrl.pathname;
  const isAuthPage = path === "/login" || path === "/signup";

  const requiresAuth =
    matchesRoute(path, signedInOnly) ||
    matchesRoute(path, coachOnly) ||
    matchesRoute(path, analystOnly);

  let user: Awaited<ReturnType<typeof supabase.auth.getUser>>["data"]["user"] = null;
  try {
    const {
      data: { user: resolvedUser },
    } = await supabase.auth.getUser();
    user = resolvedUser;
  } catch {
    user = null;
  }

  if (!user && requiresAuth) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", `${path}${request.nextUrl.search}`);
    return NextResponse.redirect(url);
  }

  if (!user) {
    return response;
  }

  let profile: { role?: UserRole } | null = null;
  try {
    const { data } = await supabase
      .from("profiles")
      .select("role")
      .eq("id", user.id)
      .single();
    profile = data as { role?: UserRole } | null;
  } catch {
    profile = null;
  }

  const metadataRole =
    user.user_metadata?.role === "coach" || user.user_metadata?.role === "analyst"
      ? (user.user_metadata.role as UserRole)
      : null;

  const role = ((profile?.role as UserRole | undefined) ?? metadataRole ?? null) as UserRole;

  if (isAuthPage) {
    const url = request.nextUrl.clone();
    url.pathname = defaultPathForRole(role);
    url.searchParams.delete("next");
    return NextResponse.redirect(url);
  }

  if (matchesRoute(path, coachOnly) && role !== "coach") {
    return NextResponse.redirect(new URL(defaultPathForRole(role), request.url));
  }

  if (matchesRoute(path, analystOnly) && role !== "analyst") {
    return NextResponse.redirect(new URL(defaultPathForRole(role), request.url));
  }

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)"],
};