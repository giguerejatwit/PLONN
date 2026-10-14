# Referral entry: local validation and release workflow

Run from `/Users/indoorgeo/Dev/PLONN`:

```sh
bash app/ios/Tests/run_referrals.sh
bash app/backend/tests/run_referrals.sh
xcodebuild -project app/ios/PLONN.xcodeproj -scheme PLONN \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/plonn-referral-build CODE_SIGNING_ALLOWED=NO build
```

The Swift runner copies only Referral.swift and its XCTest suite into a temporary Swift package. It uses `/tmp/plonn-referral-swift-build` and leaves the app's existing package/build alone. The SQL runner requires Docker; it applies the current migration stack to a disposable PostgreSQL 17 container with no published ports, extends a minimal Auth test stub, tests permissions/lifecycle, and submits competing RPCs from two sessions. No cloud credentials are read. Containers and the temporary Swift package are removed on exit.

To run the signup UI smoke test, choose a **fresh disposable simulator** (no signed-in account) and use its UDID:

```sh
xcodebuild -project app/ios/PLONN.xcodeproj -scheme PLONN \
  -destination 'platform=iOS Simulator,id=YOUR_DISPOSABLE_SIMULATOR_UDID' \
  -derivedDataPath /tmp/plonn-referral-build \
  -only-testing:PLONNUITests/PLONNUITests/testReferralSignupEntryIsOptionalAndClearsWithMode \
  CODE_SIGNING_ALLOWED=NO test
```

This UI test navigates to Account, verifies the optional signup input, fills only local form fields, and verifies the draft clears when switching auth mode. It never submits signup. Do not run other Xcode builds against this derived-data directory concurrently.

## Verified in this implementation run

- Simulator Debug build succeeded using `/tmp/plonn-referral-build`.
- Seven isolated Swift XCTest cases passed: normalization, confirmed-only applied status, idempotency, signout/account-switch isolation, failure/retry, malformed success, and wrong-owner reply handling.
- Disposable PostgreSQL suite passed against the current full migration stack, including anonymous/direct-write denial, owner isolation, inactive/expired/invalid codes, pending verification, metadata preservation/cleanup, duplicate/replacement submissions, account deletion, and competing sessions.
- The referral signup UI smoke test passed on a separate iOS 26.5 simulator, which was removed afterward. Result bundle: `/tmp/plonn-referral-build/Logs/Test/Test-PLONN-2026.09.30_15-53-12--0400.xcresult`.
- Existing unrelated UI tests still reference older tab names in their shared navigation helper. The new referral test uses the current settings accessibility label and waits for the launch intro to finish; it does not edit unrelated test behavior.

## Behavior and trust boundary

- Signup sends optional normalized input as `pending_referral_code` in Supabase Auth user metadata. Supabase retains it on that account across app restarts, signout, and email verification; no device-global referral preference is used. A signup response without a session reports that the code is **not applied**. Duplicate signup/obfuscated Auth responses are never treated as authenticated users.
- On an authenticated owner change, AccountStore calls `apply_referral_code` with that owner as a race guard. With no explicit code, the RPC reads that same owner's pending metadata or saved attribution. It requires a confirmed email and checks the registered code's active/expiry state before the first insert. Metadata is untrusted input, never attribution authority.
- Settings > Account & sync includes entry for a missed code, status refresh/retry, and the applied code. Errors do not prevent account creation or sign-in.
- A row lock serializes first attribution. One row per user is enforced by the primary key. Duplicate submissions preserve the original attribution; a different subsequent code does not replace it. Existing attribution remains valid as history if the code later expires or is deactivated.
- Clients cannot list/write the registry or insert/update/delete attribution directly. They can read only their own attribution. The definer RPC has an empty search path, qualified objects, authenticated-only execution, and an `auth.uid()` check. Attribution snapshots the registered affiliate ID. Account deletion cascades attribution; the registry represents administratively managed affiliates independently of app accounts.
- UI request tokens discard late responses and failures after signout or A → B → A transitions. Drafts reset on account/mode changes. No discount, pricing, checkout, commission, or entitlement behavior changes.

## Deployment requirement and remaining integration check

**The migration is prepared, not deployed to production.** Before enabling successful hosted entry, an authorized operator must apply `app/backend/supabase/migrations/20260930000700_referral_attribution.sql` through the normal deployment process and register actual affiliate codes using trusted administration/service-role credentials. No production codes or discounts have been invented. Registry codes are 1–64 ASCII letters/digits/underscore/hyphen, start with a letter/digit, and are stored uppercase. Never ship service-role credentials in the app.

The local SQL stub validates the database contract, not Supabase's email delivery or a real GoTrue signup round trip. After authorized deployment to an isolated Supabase test project, verify with disposable accounts:

1. Enable email confirmation; sign up with an active code. Before confirmation there must be no attribution and the UI must not claim applied.
2. Close/reopen the app; sign in as another account and verify no pending code leaks. Confirm the original email and sign in: the correct code should apply once. Repeat via the auth deep link as well as password sign-in.
3. Create an account without a code; add one in Account & sync. Try invalid, inactive, expired, mixed-case/whitespace, duplicate and replacement codes. Verify immutable owner-scoped database rows.
4. Fail/disconnect an RPC and retry; sign out or switch accounts while it is in flight. Verify no stale status or draft appears for another account. Restart and refresh to recover server status.
5. Delete the disposable accounts and verify attribution cleanup.

No successful live attribution is asserted by these local tests. Parent task owns backlog updates and the final integrated review.

## Files in this change

- `app/ios/PLONN/Account.swift`: signup metadata, referral RPC integration, Account UI.
- `app/ios/PLONN/Referral.swift`: normalized input and owner/request-scoped presentation state.
- `app/ios/PLONN.xcodeproj/project.pbxproj`: register Referral.swift in the existing app target only.
- `app/ios/Tests/ReferralStateTests.swift`, `run_referrals.sh`, this workflow.
- `app/ios/UITests/PLONNUITests.swift`: one added referral-only smoke test; existing tests preserved.
- `app/backend/supabase/migrations/20260930000700_referral_attribution.sql`.
- `app/backend/tests/referrals.sql`, `run_referrals.sh`.

No edits to PLONNApp.swift, AILab.swift, ModelSetup.swift, or backend/local_lab are part of this change.
