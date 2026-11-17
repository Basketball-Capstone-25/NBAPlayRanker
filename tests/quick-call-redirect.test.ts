import { afterEach, describe, expect, it } from "vitest";
import nextConfig from "../next.config.mjs";

describe("old Quick-Call links (SCRUM-795)", () => {
  const env = process.env.VERCEL_ENV;
  afterEach(() => {
    process.env.VERCEL_ENV = env;
  });

  it("redirect to Context ML locally and in production", async () => {
    for (const value of ["", "production"]) {
      process.env.VERCEL_ENV = value;
      const rules = await nextConfig.redirects();
      expect(rules[0]).toEqual({ source: "/context/quick-call", destination: "/context", permanent: true });
    }
  });

  it("keeps the production host redirect after it", async () => {
    process.env.VERCEL_ENV = "production";
    const rules = await nextConfig.redirects();
    expect(rules).toHaveLength(2);
    expect(rules[1].destination).toBe("https://nbaplayranker-seven.vercel.app/:path*");
  });
});
