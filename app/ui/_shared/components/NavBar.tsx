"use client";

// UI Layer — the site header: workspace links for the signed-in role, Account and Sign out.
// Links are filtered by role so coaches and analysts only see pages they can open.

import "./navigation.css";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createClient } from "../../../../lib/supabase/client";

type UserRole = "coach" | "analyst" | null;

type NavItem = {
  href: string;
  label: string;
  accent?: boolean;
};

/** Top navigation with desktop and mobile layouts; reads the session and role from Supabase. */
export default function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const supabase = useMemo(() => createClient(), []);

  const headerRef = useRef<HTMLElement | null>(null);
  const shotMenuRef = useRef<HTMLDetailsElement | null>(null);
  const menuButtonRef = useRef<HTMLButtonElement | null>(null);

  const [mobileOpen, setMobileOpen] = useState(false);
  const [authLoading, setAuthLoading] = useState(true);
  const [userRole, setUserRole] = useState<UserRole>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);

  const publicPrimaryLinks = useMemo<NavItem[]>(() => [{ href: "/", label: "Home" }], []);
  const signedInPrimaryLinks = useMemo<NavItem[]>(
    () => [
      { href: "/", label: "Home" },
      { href: "/glossary", label: "Glossary" },
    ],
    []
  );

  const coachWorkspaceLinks = useMemo<NavItem[]>(
    () => [
      { href: "/matchup", label: "Baseline" },
      { href: "/matchup-dashboard", label: "Matchup dashboard" },
      { href: "/context", label: "Game context" },
      { href: "/gameplan", label: "Gameplan" },
    ],
    []
  );

  const analystWorkspaceLinks = useMemo<NavItem[]>(
    () => [
      { href: "/data-explorer", label: "Data Explorer" },
      { href: "/model-metrics", label: "Model Metrics" },
      { href: "/statistical-analysis", label: "Statistical Analysis" },
      { href: "/request-log", label: "Request Log" },
    ],
    []
  );

  const shotLinks: NavItem[] = [
    { href: "/shot-explorer", label: "Shot Explorer" },
    { href: "/shot-heatmap", label: "Shot Heatmap" },
    { href: "/shot-plan", label: "Shot Plan" },
    { href: "/shot-model-metrics", label: "Shot Model Metrics" },
    { href: "/shot-statistical-analysis", label: "Shot Statistical Analysis" },
  ];

  const primaryLinks = userEmail ? signedInPrimaryLinks : publicPrimaryLinks;

  const workspaceLinks = useMemo<NavItem[]>(() => {
    if (userRole === "coach") return coachWorkspaceLinks;
    if (userRole === "analyst") return analystWorkspaceLinks;
    return [];
  }, [userRole, coachWorkspaceLinks, analystWorkspaceLinks]);

  const workspaceLabel =
    userRole === "coach"
      ? "Coach workspace"
      : userRole === "analyst"
      ? "Analyst workspace"
      : null;

  function isActive(href: string) {
    if (href === "/") return pathname === "/";
    return pathname === href || pathname?.startsWith(`${href}/`);
  }

  function closeAll() {
    setMobileOpen(false);
    if (shotMenuRef.current) shotMenuRef.current.open = false;
  }

  async function loadProfile() {
    setAuthLoading(true);

    const {
      data: { user },
      error,
    } = await supabase.auth.getUser();

    if (error || !user) {
      setUserEmail(null);
      setUserRole(null);
      setAuthLoading(false);
      return;
    }

    setUserEmail(user.email ?? null);

    const { data: profile } = await supabase
      .from("profiles")
      .select("role")
      .eq("id", user.id)
      .single();

    setUserRole(profile?.role === "coach" || profile?.role === "analyst" ? profile.role : null);
    setAuthLoading(false);
  }

  useEffect(() => {
    let mounted = true;

    const init = async () => {
      if (!mounted) return;
      await loadProfile();
    };

    init();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(() => {
      if (!mounted) return;
      loadProfile();
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [supabase]);

  useEffect(() => {
    closeAll();
  }, [pathname]);

  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      const target = e.target as Node;
      const inside = headerRef.current?.contains(target);
      if (!inside) closeAll();
    }

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        const mobilePanel = headerRef.current?.querySelector("#workspace-menu.open");
        if (mobilePanel?.contains(document.activeElement)) menuButtonRef.current?.focus();
        if (shotMenuRef.current?.open && shotMenuRef.current.contains(document.activeElement)) {
          shotMenuRef.current.querySelector("summary")?.focus();
        }
        closeAll();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  async function handleSignOut() {
    await supabase.auth.signOut();
    setUserEmail(null);
    setUserRole(null);
    setMobileOpen(false);
    router.push("/");
    router.refresh();
  }

  function NavLink({
    item,
    mobile = false,
    workspace = false,
  }: {
    item: NavItem;
    mobile?: boolean;
    workspace?: boolean;
  }) {
    const active = isActive(item.href);

    const className = [
      "npr-link",
      mobile ? "npr-link-mobile" : "",
      workspace ? "npr-link-workspace" : "",
      item.accent ? "npr-link-accent" : "",
      active ? "npr-link-active" : "",
    ]
      .filter(Boolean)
      .join(" ");

    return (
      <Link
        href={item.href}
        className={className}
        aria-current={active ? "page" : undefined}
        onClick={closeAll}
      >
        {item.label}
      </Link>
    );
  }

  function DesktopAuth() {
    if (authLoading) {
      return <div className="npr-loading-chip">Loading…</div>;
    }

    if (!userEmail) {
      return (
        <div className="npr-auth-combo">
          <Link
            href="/signup"
            className="npr-auth-combo-link npr-auth-combo-link-primary"
            onClick={closeAll}
          >
            Sign up
          </Link>
          <span className="npr-auth-combo-divider" aria-hidden>
            /
          </span>
          <Link href="/login" className="npr-auth-combo-link" onClick={closeAll}>
            Log in
          </Link>
        </div>
      );
    }

    return (
      <div className="npr-auth-group">
        <Link href="/account" className="npr-auth-combo-link" onClick={closeAll}>
          Account
        </Link>
        <button type="button" className="npr-cta-signout" onClick={handleSignOut}>
          Sign out
        </button>
      </div>
    );
  }

  function MobileAuth() {
    if (authLoading) {
      return <div className="npr-mobile-meta">Loading account…</div>;
    }

    if (!userEmail) {
      return (
        <div className="npr-mobile-auth">
          <Link
            href="/signup"
            className="npr-auth-link-mobile npr-auth-link-mobile-primary"
            onClick={closeAll}
          >
            Sign up
          </Link>
          <Link href="/login" className="npr-auth-link-mobile" onClick={closeAll}>
            Log in
          </Link>
        </div>
      );
    }

    return (
      <div className="npr-mobile-auth">
        <Link href="/account" className="npr-auth-link-mobile" onClick={closeAll}>
          Account
        </Link>
        <button type="button" className="npr-mobile-cta-signout" onClick={handleSignOut}>
          Sign out
        </button>
      </div>
    );
  }

  return (
    <header ref={headerRef} className="npr-header">
      <div className="npr-shell">
        <div className="npr-topbar">
          <div className="npr-brand">
            <Link href="/" className="npr-brand-wrap" onClick={closeAll}>
              <span className="npr-brand-row">
                <span className="npr-mark" aria-hidden>
                  <svg className="npr-mark-icon" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.94)" strokeWidth="1.8" />
                    <path d="M12 3v18" stroke="rgba(255,255,255,0.94)" strokeWidth="1.4" strokeLinecap="round" />
                    <path d="M3 12h18" stroke="rgba(255,255,255,0.94)" strokeWidth="1.4" strokeLinecap="round" />
                    <path
                      d="M6.2 5.2c2.8 3 4.1 5.6 4.1 6.8s-1.3 3.8-4.1 6.8"
                      stroke="rgba(255,255,255,0.94)"
                      strokeWidth="1.4"
                      strokeLinecap="round"
                    />
                    <path
                      d="M17.8 5.2c-2.8 3-4.1 5.6-4.1 6.8s1.3 3.8 4.1 6.8"
                      stroke="rgba(255,255,255,0.94)"
                      strokeWidth="1.4"
                      strokeLinecap="round"
                    />
                  </svg>
                </span>

                <span className="npr-brand-copy">
                  <span className="npr-brand-title">NBA Play Ranker</span>
                  <span className="npr-brand-subtitle">Basketball decision support</span>
                </span>
              </span>
            </Link>
          </div>

          <div className="npr-desktop">
            <nav className="npr-primary-nav" aria-label="Primary navigation">
              {primaryLinks.map((item) => (
                <NavLink key={item.href} item={item} />
              ))}
            </nav>

            <DesktopAuth />
          </div>

          <button
            ref={menuButtonRef}
            type="button"
            className={`npr-burger ${mobileOpen ? "open" : ""}`}
            aria-label="Toggle menu"
            aria-expanded={mobileOpen}
            aria-controls="workspace-menu"
            onClick={() => setMobileOpen((v) => !v)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>

        {userEmail && workspaceLinks.length > 0 ? (
          <div className="npr-workspace-row">
            <div className="npr-workspace-pill">
              <span className="npr-workspace-pill-dot" aria-hidden />
              {workspaceLabel}
            </div>

            <nav className="npr-workspace-scroll" aria-label="Workspace navigation">
              {workspaceLinks.map((item) => (
                <NavLink key={item.href} item={item} workspace />
              ))}
            </nav>
            {userRole === "analyst" ? (
              <details className="npr-shots" ref={shotMenuRef}>
                <summary className={pathname?.startsWith("/shot-") ? "npr-link-active" : ""}>Shot tools <span aria-hidden>⌄</span></summary>
                <nav aria-label="Shot tools">{shotLinks.map(item => <NavLink key={item.href} item={item} />)}</nav>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>

      <div id="workspace-menu" className={`npr-mobile-panel ${mobileOpen ? "open" : ""}`}>
        <div className="npr-mobile-section">
          <div className="npr-mobile-heading">Navigation</div>
          {primaryLinks.map((item) => (
            <NavLink key={item.href} item={item} mobile />
          ))}
        </div>

        {userEmail && workspaceLinks.length > 0 ? (
          <div className="npr-mobile-section">
            <div className="npr-mobile-heading">{workspaceLabel}</div>
            {[...workspaceLinks, ...(userRole === "analyst" ? shotLinks : [])].map((item) => (
              <NavLink key={item.href} item={item} mobile workspace />
            ))}
          </div>
        ) : null}

        <div className="npr-mobile-section">
          <div className="npr-mobile-heading">Account</div>
          <MobileAuth />
        </div>
      </div>

    </header>
  );
}
