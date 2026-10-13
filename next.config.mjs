import { fileURLToPath } from "node:url";

export default {
  reactStrictMode: true,
  outputFileTracingRoot: fileURLToPath(new URL(".", import.meta.url)),
  async redirects() {
    if (process.env.VERCEL_ENV !== "production") return [];

    // Keep sign-in cookies and browser API requests on the approved public origin.
    return [{
      source: "/:path*",
      missing: [{ type: "host", value: "nbaplayranker-seven\\.vercel\\.app" }],
      destination: "https://nbaplayranker-seven.vercel.app/:path*",
      permanent: false,
    }];
  },
};
