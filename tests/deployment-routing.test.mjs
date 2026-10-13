import { afterEach, describe, expect, it, vi } from "vitest";
import { getRedirectUrl, unstable_getResponseFromNextConfig } from "next/experimental/testing/server";
import nextConfig from "../next.config.mjs";

const publicOrigin = "https://nbaplayranker-seven.vercel.app";
const buildOrigin = "https://nbaplayranker-bxg9zds45-pearson-manufacturing.vercel.app";
const request = (url) => unstable_getResponseFromNextConfig({ url, nextConfig });

afterEach(() => vi.unstubAllEnvs());

describe("production deployment links", () => {
  it.each([
    "/",
    "/data-explorer?season=2024-25&team=TOR",
    "/login?next=%2Fdata-explorer",
    "/api/exports?path=%2Fdata%2Fteam-playtypes.csv&query=season%3D2024-25",
  ])("keeps the path and query when redirecting %s to the public site", async (path) => {
    vi.stubEnv("VERCEL_ENV", "production");
    const response = await request(buildOrigin + path);
    expect(response.status).toBe(307);
    expect(getRedirectUrl(response)).toBe(publicOrigin + path);
  });

  it("does not loop on the public site", async () => {
    vi.stubEnv("VERCEL_ENV", "production");
    expect(getRedirectUrl(await request(publicOrigin + "/data-explorer"))).toBeNull();
  });

  it.each(["preview", "development", ""])("leaves %s deployments in their own environment", async (environment) => {
    vi.stubEnv("VERCEL_ENV", environment);
    expect(getRedirectUrl(await request(buildOrigin + "/data-explorer"))).toBeNull();
    expect(getRedirectUrl(await request("http://localhost:3000/data-explorer"))).toBeNull();
  });

  it("does not treat a lookalike host as the public site or accept a supplied destination", async () => {
    vi.stubEnv("VERCEL_ENV", "production");
    const response = await request("https://nbaplayranker-sevenXvercelXapp/login?next=https%3A%2F%2Fother.example");
    expect(new URL(getRedirectUrl(response)).origin).toBe(publicOrigin);
  });
});
