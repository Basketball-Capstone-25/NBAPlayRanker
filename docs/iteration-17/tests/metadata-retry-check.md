# Metadata retry check

BST-194 uses the actual Data Explorer metadata panel, API request helper and response parser. The fixture creates small temporary source files and reads them with the backend catalog. It does not use production data or credentials.

Start it from the repository root:

```sh
node scripts/metadata-retry-harness.cjs --out /tmp/metadata-browser-check
```

Open the local URL printed by the script. The fixture checks missing shot files, a corrupt shot file and an invalid source-date sidecar before starting. A valid three-row play-type source remains available in each case.

1. On `service-error`, confirm the Data source panel remains visible and shows an unavailable message with Retry. Click Retry; the source, two seasons and three rows should appear.
2. Open `invalid-json` and repeat the retry check.
3. Open `invalid-schema` and repeat the retry check.
4. Open `corrupt-source`. The valid play-type source should still appear. The fixture's catalog records the shot source as unavailable.
5. Open `malformed-sidecar`. The source should still appear. Expand Source details and check that the original pull date says it was not recorded.

Each of the first three scenarios fails only on its first request. Restart the script to repeat the initial failures. Component states and responses are saved to `events.json` in the selected output folder. Browser results should be recorded only after these steps are run.

Authentication is outside this fixture: the session lookup returns no session, and Next Link uses an ordinary local anchor. A content security policy limits requests to the local fixture. Production authentication has separate checks.
