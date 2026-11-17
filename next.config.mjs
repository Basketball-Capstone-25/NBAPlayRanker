import { fileURLToPath } from "node:url";

export default {
  reactStrictMode: true,
  outputFileTracingRoot: fileURLToPath(new URL(".", import.meta.url)),
  async redirects() {
    // SCRUM-795: Quick-Call Mode was folded into Context ML. Old links keep
    // their query string, so the situation is still filled in.
    const quickCall = { source: "/context/quick-call", destination: "/context", permanent: true };
    if (process.env.VERCEL_ENV !== "production") return [quickCall];

    // Keep sign-in cookies and browser API requests on the approved public origin.
    return [quickCall, {
      source: "/:path*",
      missing: [{ type: "host", value: "nbaplayranker-seven\\.vercel\\.app" }],
      destination: "https://nbaplayranker-seven.vercel.app/:path*",
      permanent: false,
    }];
  },
};
