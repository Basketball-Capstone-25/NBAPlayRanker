"use client";

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

export default function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const supabase = useMemo(() => createClient(), []);

  const headerRef = useRef<HTMLElement | null>(null);

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
      { href: "/matchup", label: "Matchup / Baseline" },
      { href: "/context", label: "Context / ML" },
      { href: "/gameplan", label: "Gameplan", accent: true },
    ],
    []
  );

  const analystWorkspaceLinks = useMemo<NavItem[]>(
    () => [
      { href: "/data-explorer", label: "Data Explorer" },
      { href: "/model-metrics", label: "Model Metrics" },
      { href: "/statistical-analysis", label: "Statistical Analysis" },
      { href: "/shot-explorer", label: "Shot Explorer" },
      { href: "/shot-heatmap", label: "Shot Heatmap" },
      { href: "/shot-plan", label: "Shot Plan" },
      { href: "/shot-model-metrics", label: "Shot Model Metrics" },
      { href: "/shot-statistical-analysis", label: "Shot Statistical Analysis" },
    ],
    []
  );

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

    const metadataRole =
      user.user_metadata?.role === "coach" || user.user_metadata?.role === "analyst"
        ? (user.user_metadata.role as UserRole)
        : null;

    setUserRole(((profile?.role as UserRole | undefined) ?? metadataRole ?? null) as UserRole);
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
    setMobileOpen(false);
  }, [pathname]);

  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      const target = e.target as Node;
      const inside = headerRef.current?.contains(target);
      if (!inside) setMobileOpen(false);
    }

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setMobileOpen(false);
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
                  <span className="npr-brand-subtitle">Decision Support • Explainable + AI</span>
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
            type="button"
            className={`npr-burger ${mobileOpen ? "open" : ""}`}
            aria-label="Toggle menu"
            aria-expanded={mobileOpen}
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
          </div>
        ) : null}
      </div>

      <div className={`npr-mobile-panel ${mobileOpen ? "open" : ""}`}>
        <div className="npr-mobile-section">
          <div className="npr-mobile-heading">Navigation</div>
          {primaryLinks.map((item) => (
            <NavLink key={item.href} item={item} mobile />
          ))}
        </div>

        {userEmail && workspaceLinks.length > 0 ? (
          <div className="npr-mobile-section">
            <div className="npr-mobile-heading">{workspaceLabel}</div>
            {workspaceLinks.map((item) => (
              <NavLink key={item.href} item={item} mobile workspace />
            ))}
          </div>
        ) : null}

        <div className="npr-mobile-section">
          <div className="npr-mobile-heading">Account</div>
          <MobileAuth />
        </div>
      </div>

      <style jsx>{`
        .npr-header {
          position: sticky;
          top: 0;
          z-index: 60;
          background: rgba(255, 255, 255, 0.78);
          backdrop-filter: blur(18px);
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
          box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
        }

        .npr-header::before {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
          background:
            radial-gradient(760px 120px at 12% -30%, rgba(37, 99, 235, 0.12), transparent 62%),
            radial-gradient(760px 120px at 88% -30%, rgba(124, 58, 237, 0.12), transparent 62%),
            radial-gradient(760px 180px at 80% 120%, rgba(236, 72, 153, 0.08), transparent 62%);
          opacity: 0.92;
        }

        .npr-shell {
          position: relative;
          max-width: 1220px;
          margin: 0 auto;
          padding: 8px 16px 10px;
        }

        .npr-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 18px;
          min-height: 56px;
        }

        .npr-brand {
          min-width: 0;
          flex: 0 0 auto;
        }

        .npr-brand-wrap {
          display: inline-block !important;
          text-decoration: none !important;
          min-width: 0;
        }

        .npr-brand-row {
          display: flex !important;
          flex-direction: row !important;
          flex-wrap: nowrap !important;
          align-items: center !important;
          justify-content: flex-start !important;
          gap: 12px !important;
          min-width: 0;
          white-space: nowrap;
        }

        .npr-mark {
          width: 42px;
          height: 42px;
          border-radius: 14px;
          background:
            radial-gradient(circle at 34% 28%, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.5) 34%, transparent 68%),
            linear-gradient(135deg, #2563eb 0%, #7c3aed 56%, #ec4899 100%);
          display: grid;
          place-items: center;
          border: 1px solid rgba(255,255,255,0.36);
          box-shadow: 0 14px 28px rgba(37, 99, 235, 0.18);
          flex: 0 0 42px;
        }

        .npr-mark-icon {
          width: 31px;
          height: 31px;
          filter: drop-shadow(0 2px 6px rgba(15, 23, 42, 0.2));
        }

        .npr-brand-copy {
          display: flex !important;
          flex-direction: column !important;
          justify-content: center !important;
          align-items: flex-start !important;
          gap: 3px;
          min-width: 0;
          white-space: nowrap;
          flex: 0 1 auto;
        }

        .npr-brand-title {
          font-size: 17px;
          line-height: 1;
          font-weight: 950;
          letter-spacing: -0.045em;
          color: #0f172a;
          text-decoration: none !important;
        }

        .npr-brand-subtitle {
          font-size: 12px;
          line-height: 1.15;
          letter-spacing: -0.01em;
          color: rgba(15, 23, 42, 0.6);
          white-space: nowrap;
          font-weight: 600;
          text-decoration: none !important;
        }

        .npr-desktop {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 12px;
          min-width: 0;
          flex: 1 1 auto;
        }

        .npr-primary-nav {
          display: flex;
          align-items: center;
          gap: 8px;
          min-width: 0;
        }

        .npr-link {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          text-decoration: none !important;
          white-space: nowrap;
          border-radius: 999px;
          padding: 9px 14px;
          border: 1px solid transparent;
          background: rgba(255, 255, 255, 0.5);
          color: rgba(15, 23, 42, 0.82);
          font-size: 14px;
          font-weight: 850;
          letter-spacing: -0.02em;
          line-height: 1;
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
          transition:
            transform 160ms ease,
            background 160ms ease,
            border-color 160ms ease,
            box-shadow 160ms ease,
            color 160ms ease;
        }

        .npr-link:hover {
          transform: translateY(-1px);
          background: rgba(255, 255, 255, 0.88);
          border-color: rgba(37, 99, 235, 0.16);
          color: rgba(15, 23, 42, 0.94);
          box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
        }

        .npr-link-active {
          background: rgba(37, 99, 235, 0.08);
          border-color: rgba(37, 99, 235, 0.18);
          color: #0f172a;
          box-shadow: 0 14px 28px rgba(37, 99, 235, 0.1);
        }

        .npr-link-workspace {
          background: rgba(255, 255, 255, 0.62);
          border-color: rgba(15, 23, 42, 0.06);
          padding: 8px 13px;
          font-size: 14px;
        }

        .npr-link-accent {
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.22);
          box-shadow: 0 16px 34px rgba(124, 58, 237, 0.2);
        }

        .npr-link-accent:hover {
          border-color: rgba(255, 255, 255, 0.36);
          color: #ffffff;
          filter: brightness(1.03);
        }

        .npr-link-accent.npr-link-active {
          box-shadow: 0 18px 38px rgba(124, 58, 237, 0.26);
          outline: 2px solid rgba(255, 255, 255, 0.48);
          outline-offset: 2px;
        }

        .npr-auth-group {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-shrink: 0;
        }

        .npr-auth-combo {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 5px 7px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background:
            linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.72)),
            rgba(255,255,255,0.78);
          box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
          backdrop-filter: blur(10px);
        }

        .npr-auth-combo-link {
          text-decoration: none !important;
          color: rgba(15, 23, 42, 0.82);
          font-size: 14px;
          font-weight: 850;
          letter-spacing: -0.02em;
          line-height: 1;
          padding: 9px 12px;
          border-radius: 999px;
          transition: background 160ms ease, color 160ms ease, transform 160ms ease;
        }

        .npr-auth-combo-link:hover {
          background: rgba(15, 23, 42, 0.04);
          color: rgba(15, 23, 42, 0.96);
          transform: translateY(-1px);
        }

        .npr-auth-combo-link-primary {
          background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.12) 0%,
            rgba(124, 58, 237, 0.12) 55%,
            rgba(236, 72, 153, 0.12) 100%
          );
          color: #0f172a;
        }

        .npr-auth-combo-divider {
          color: rgba(15, 23, 42, 0.34);
          font-weight: 700;
          user-select: none;
        }

        .npr-loading-chip {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 92px;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(255, 255, 255, 0.68);
          color: rgba(15, 23, 42, 0.62);
          font-size: 13px;
          font-weight: 800;
          box-shadow: 0 10px 20px rgba(15, 23, 42, 0.04);
        }

        .npr-cta-signout {
          appearance: none;
          border: 1px solid rgba(255, 255, 255, 0.18);
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          color: #ffffff;
          padding: 12px 20px;
          min-height: 46px;
          border-radius: 18px;
          font-size: 14px;
          font-weight: 900;
          letter-spacing: -0.02em;
          line-height: 1;
          cursor: pointer;
          box-shadow: 0 18px 34px rgba(124, 58, 237, 0.2);
          transition:
            transform 160ms ease,
            filter 160ms ease,
            box-shadow 160ms ease,
            border-color 160ms ease;
        }

        .npr-cta-signout:hover {
          transform: translateY(-1px);
          filter: brightness(1.03);
          border-color: rgba(255, 255, 255, 0.28);
          box-shadow: 0 22px 40px rgba(124, 58, 237, 0.26);
        }

        .npr-workspace-row {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(15, 23, 42, 0.07);
          min-width: 0;
        }

        .npr-workspace-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 9px 13px;
          border-radius: 999px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(255, 255, 255, 0.68);
          color: rgba(15, 23, 42, 0.84);
          font-size: 12px;
          font-weight: 900;
          white-space: nowrap;
          box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
          flex-shrink: 0;
        }

        .npr-workspace-pill-dot {
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.08);
        }

        .npr-workspace-scroll {
          display: flex;
          align-items: center;
          gap: 8px;
          overflow-x: auto;
          padding-bottom: 2px;
          scrollbar-width: none;
          min-width: 0;
          flex: 1 1 auto;
        }

        .npr-workspace-scroll::-webkit-scrollbar {
          display: none;
        }

        .npr-burger {
          display: none;
          width: 42px;
          height: 40px;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.1);
          background: rgba(255, 255, 255, 0.86);
          box-shadow: 0 10px 20px rgba(15, 23, 42, 0.05);
          cursor: pointer;
          padding: 0;
          position: relative;
          flex-shrink: 0;
        }

        .npr-burger span {
          position: absolute;
          left: 12px;
          right: 12px;
          height: 2px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.86);
          transition: transform 180ms ease, opacity 180ms ease, top 180ms ease;
        }

        .npr-burger span:nth-child(1) {
          top: 13px;
        }

        .npr-burger span:nth-child(2) {
          top: 19px;
        }

        .npr-burger span:nth-child(3) {
          top: 25px;
        }

        .npr-burger.open span:nth-child(1) {
          top: 19px;
          transform: rotate(45deg);
        }

        .npr-burger.open span:nth-child(2) {
          opacity: 0;
        }

        .npr-burger.open span:nth-child(3) {
          top: 19px;
          transform: rotate(-45deg);
        }

        .npr-mobile-panel {
          display: none;
          max-width: 1220px;
          margin: 0 auto;
          padding: 0 16px 14px;
        }

        .npr-mobile-panel.open {
          display: block;
        }

        .npr-mobile-section {
          margin-top: 10px;
          padding: 14px;
          border-radius: 20px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(255, 255, 255, 0.8);
          box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
        }

        .npr-mobile-heading {
          font-size: 11px;
          font-weight: 900;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: rgba(15, 23, 42, 0.48);
          margin-bottom: 10px;
        }

        .npr-link-mobile {
          display: flex;
          justify-content: flex-start;
          width: 100%;
          margin-top: 8px;
          border-radius: 16px;
          padding: 13px 14px;
        }

        .npr-mobile-auth {
          display: grid;
          gap: 10px;
        }

        .npr-auth-link-mobile {
          appearance: none;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(255, 255, 255, 0.84);
          color: rgba(15, 23, 42, 0.88);
          text-decoration: none !important;
          display: inline-flex;
          justify-content: center;
          align-items: center;
          width: 100%;
          border-radius: 16px;
          padding: 13px 14px;
          font-size: 14px;
          font-weight: 850;
          letter-spacing: -0.02em;
          box-shadow: 0 12px 26px rgba(15, 23, 42, 0.04);
          cursor: pointer;
        }

        .npr-auth-link-mobile-primary {
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.18);
          box-shadow: 0 18px 34px rgba(124, 58, 237, 0.2);
        }

        .npr-mobile-cta-signout {
          appearance: none;
          width: 100%;
          min-height: 52px;
          border: 1px solid rgba(255, 255, 255, 0.18);
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
          color: #ffffff;
          border-radius: 18px;
          padding: 14px 16px;
          font-size: 14px;
          font-weight: 900;
          letter-spacing: -0.02em;
          box-shadow: 0 18px 34px rgba(124, 58, 237, 0.2);
          cursor: pointer;
        }

        .npr-mobile-meta {
          padding: 13px 14px;
          border-radius: 16px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: rgba(15, 23, 42, 0.03);
          color: rgba(15, 23, 42, 0.78);
          font-weight: 800;
        }

        @media (max-width: 1120px) {
          .npr-link,
          .npr-link-workspace,
          .npr-auth-combo-link,
          .npr-cta-signout {
            font-size: 13.5px;
          }

          .npr-primary-nav,
          .npr-workspace-scroll {
            gap: 7px;
          }
        }

        @media (max-width: 980px) {
          .npr-desktop {
            display: none;
          }

          .npr-workspace-row {
            display: none;
          }

          .npr-burger {
            display: inline-block;
          }
        }

        @media (max-width: 640px) {
          .npr-shell {
            padding: 8px 14px 10px;
          }

          .npr-brand-row {
            gap: 10px !important;
          }

          .npr-brand-title {
            font-size: 15px;
          }

          .npr-brand-subtitle {
            font-size: 10.5px;
            white-space: normal;
          }

          .npr-mark {
            width: 38px;
            height: 38px;
            border-radius: 13px;
            flex-basis: 38px;
          }
        }
      `}</style>
    </header>
  );
}