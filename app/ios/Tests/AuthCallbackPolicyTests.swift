import XCTest
@testable import PLONNCore

final class AuthCallbackPolicyTests: XCTestCase {
    func testAIConsentRequiresCurrentExplicitChoiceAndRejectsAnonymousUsers() {
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: nil, localChoice: nil))
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: "v0", localChoice: nil))
        XCTAssertTrue(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: "v1", localChoice: nil))
        XCTAssertTrue(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: nil, localChoice: true))
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: "v1", localChoice: false))
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: true, metadataVersion: "v1", localChoice: true))
    }

    func testAIConsentPreferenceIsVersionedAndBoundToAccount() {
        let first = UUID(), second = UUID()
        XCTAssertEqual(AccountAIConsentPolicy.preferenceKey(owner: first), AccountAIConsentPolicy.preferenceKey(owner: first))
        XCTAssertNotEqual(AccountAIConsentPolicy.preferenceKey(owner: first), AccountAIConsentPolicy.preferenceKey(owner: second))
        XCTAssertTrue(AccountAIConsentPolicy.preferenceKey(owner: first).hasPrefix("plonn.aiConsent.v1."))
    }

    func testAcknowledgedConsentRemovesOnlyMatchingPendingOverride() {
        XCTAssertTrue(AccountAIConsentPolicy.synchronizedChoice(requested: true, pending: true, metadataVersion: "v1"))
        XCTAssertTrue(AccountAIConsentPolicy.synchronizedChoice(requested: false, pending: false, metadataVersion: "declined_v1"))
        XCTAssertFalse(AccountAIConsentPolicy.synchronizedChoice(requested: true, pending: false, metadataVersion: "v1"))
        XCTAssertFalse(AccountAIConsentPolicy.synchronizedChoice(requested: false, pending: false, metadataVersion: "v1"))
        XCTAssertFalse(AccountAIConsentPolicy.synchronizedChoice(requested: true, pending: true, metadataVersion: nil))
        // After acknowledgement clears the pending override, a later remote revoke governs.
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: "declined_v1", localChoice: nil))
        // A pending or failed local revoke still overrides old metadata.
        XCTAssertFalse(AccountAIConsentPolicy.permits(isAnonymous: false, metadataVersion: "v1", localChoice: false))
    }

    func testConsentUpdateCannotOverlapAccountOrSocialTransition() {
        XCTAssertTrue(AccountAIConsentPolicy.mayBeginConsentSync(accountBusy: false, socialPending: false, consentUpdating: false))
        XCTAssertFalse(AccountAIConsentPolicy.mayBeginConsentSync(accountBusy: true, socialPending: false, consentUpdating: false))
        XCTAssertFalse(AccountAIConsentPolicy.mayBeginConsentSync(accountBusy: false, socialPending: true, consentUpdating: false))
        XCTAssertFalse(AccountAIConsentPolicy.mayBeginConsentSync(accountBusy: false, socialPending: false, consentUpdating: true))
    }

    func testCallbackPresenceReportsStructureWithoutReturningCredentials() throws {
        let cases = [
            ("", "callback_code_missing"),
            ("?code", "callback_code_empty"),
            ("?code=", "callback_code_empty"),
            ("?code=synthetic-private-value", "callback_code_present"),
            ("#code=synthetic-private-value", "callback_code_present"),
            ("?code=one&code=two", "callback_code_ambiguous"),
            ("?code=one#code=two", "callback_code_ambiguous"),
            ("?error_code=otp_expired", "callback_error"),
            ("?code=one#error_description=synthetic-private-value", "callback_error")
        ]
        for (suffix, expected) in cases {
            let url = try XCTUnwrap(URL(string: "com.plonn.preview://auth" + suffix))
            XCTAssertEqual(AuthCallbackPresence.category(for: url), expected)
            XCTAssertFalse(AuthCallbackPresence.category(for: url).contains("synthetic-private-value"))
        }
    }

    func testAuthFailureCategoriesMapOnlyKnownCodesAndStatuses() {
        let examples: [(String?, Int?, String)] = [
            ("otp_expired", 403, "link_expired"),
            ("flow_state_expired", 400, "link_expired"),
            ("flow_state_not_found", 400, "recovery_context"),
            ("bad_code_verifier", 400, "recovery_context"),
            ("over_email_send_rate_limit", 400, "rate_limit"),
            ("over_request_rate_limit", 400, "rate_limit"),
            (nil, 429, "rate_limit"),
            ("invalid_credentials", 400, "credentials"),
            (nil, 500, "auth_service"),
            ("unknown", 503, "auth_service"),
            (nil, nil, "auth_request"),
            ("unknown", 403, "auth_request")
        ]
        for (code, status, expected) in examples {
            XCTAssertEqual(AuthFailurePresentation.category(serverCode: code, httpStatus: status), expected)
        }
    }

    func testAuthFailureMessagesExplainRecoveryWithoutEchoingInputs() {
        let expired = AuthFailurePresentation.message(forCategory: "link_expired")
        XCTAssertTrue(expired.contains("expired or was already used"))
        XCTAssertFalse(expired.contains("connection"))
        let context = AuthFailurePresentation.message(forCategory: "recovery_context")
        XCTAssertTrue(context.contains("Request a new link here"))
        XCTAssertTrue(context.contains("same app on this device"))
        let arbitraryInput = "private-input-that-must-not-be-echoed"
        XCTAssertEqual(AuthFailurePresentation.category(serverCode: arbitraryInput, httpStatus: 400), "auth_request")
        XCTAssertEqual(AuthFailurePresentation.message(forCategory: arbitraryInput), AuthFailurePresentation.message(forCategory: "auth_request"))
        XCTAssertFalse(AuthFailurePresentation.message(forCategory: arbitraryInput).contains(arbitraryInput))
    }

    func testRecoveryIntentPermitsOnlyMatchingVerifiedAccountWithinOneHour() {
        let requestedAt = Date(timeIntervalSince1970: 1_800_000_000)
        let intent = PasswordRecoveryIntent(emailFingerprint: "account-fingerprint", requestedAt: requestedAt)
        for elapsed in [0.0, 1.0, 3_599.0, 3_600.0] {
            XCTAssertTrue(intent.permitsRecovery(verifiedEmailFingerprint: "account-fingerprint", now: requestedAt.addingTimeInterval(elapsed)))
        }
        for elapsed in [-1.0, 3_600.001, 7_200.0] {
            XCTAssertFalse(intent.permitsRecovery(verifiedEmailFingerprint: "account-fingerprint", now: requestedAt.addingTimeInterval(elapsed)))
        }
    }

    func testRecoveryIntentRejectsMissingEmptyAndDifferentVerifiedAccounts() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let intent = PasswordRecoveryIntent(emailFingerprint: "account-fingerprint", requestedAt: now)
        for fingerprint in [nil, "", "other-account", "ACCOUNT-FINGERPRINT"] as [String?] {
            XCTAssertFalse(intent.permitsRecovery(verifiedEmailFingerprint: fingerprint, now: now))
        }
        XCTAssertFalse(PasswordRecoveryIntent(emailFingerprint: "", requestedAt: now)
            .permitsRecovery(verifiedEmailFingerprint: "", now: now))
    }

    func testRecoveryIntentRoundTripsOnlyFingerprintAndRequestTime() throws {
        let intent = PasswordRecoveryIntent(emailFingerprint: "account-fingerprint", requestedAt: Date(timeIntervalSince1970: 1_800_000_000))
        let data = try JSONEncoder().encode(intent)
        XCTAssertEqual(try JSONDecoder().decode(PasswordRecoveryIntent.self, from: data), intent)
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(fields.keys), ["emailFingerprint", "requestedAt"])
    }

    func testProductionAndPreviewHaveSeparateRedirectsAndRejectEachOther() throws {
        let production = try XCTUnwrap(AuthCallbackPolicy(bundleIdentifier: "com.oakas.plonn"))
        let preview = try XCTUnwrap(AuthCallbackPolicy(bundleIdentifier: "com.plonn.preview"))
        XCTAssertEqual(production.redirectURL.absoluteString, "com.oakas.plonn://auth")
        XCTAssertEqual(preview.redirectURL.absoluteString, "com.plonn.preview://auth")
        XCTAssertTrue(production.accepts(production.redirectURL))
        XCTAssertTrue(preview.accepts(preview.redirectURL))
        XCTAssertFalse(production.accepts(preview.redirectURL))
        XCTAssertFalse(preview.accepts(production.redirectURL))
    }

    func testPKCEAndRecoveryPayloadsRemainForSDKVerification() throws {
        let policy = try XCTUnwrap(AuthCallbackPolicy(bundleIdentifier: "com.oakas.plonn"))
        for suffix in ["?code=test-code", "?error=access_denied&error_description=Expired", "#type=recovery", "?code=test-code#type=recovery"] {
            XCTAssertTrue(policy.accepts(try XCTUnwrap(URL(string: "com.oakas.plonn://auth" + suffix))))
        }
        XCTAssertTrue(policy.accepts(try XCTUnwrap(URL(string: "COM.OAKAS.PLONN://AUTH?code=test-code"))))
        // A routing match never treats the payload itself as a verified session.
    }

    func testUnexpectedAuthorityPathsAndOtherAppsAreRejected() throws {
        let policy = try XCTUnwrap(AuthCallbackPolicy(bundleIdentifier: "com.oakas.plonn"))
        for value in ["https://auth?code=x", "com.oakas.plonn://other?code=x",
                      "com.oakas.plonn://auth.evil.test?code=x", "com.oakas.plonn://auth:443?code=x",
                      "com.oakas.plonn://user@auth?code=x", "com.oakas.plonn://user:password@auth?code=x",
                      "com.oakas.plonn://auth/recovery?code=x", "com.oakas.plonn://auth/?code=x",
                      "com.oakas.plonn:/auth?code=x", "com.oakas.plonn.evil://auth?code=x"] {
            XCTAssertFalse(policy.accepts(try XCTUnwrap(URL(string: value))), value)
        }
    }

    func testUnknownOrMissingAppIdentityCannotFallBackToPreview() {
        for value in [nil, "", "com.oakas.plonn.uitests", "com.example.app", "$(PRODUCT_BUNDLE_IDENTIFIER)"] as [String?] {
            XCTAssertNil(AuthCallbackPolicy(bundleIdentifier: value))
        }
    }
}
