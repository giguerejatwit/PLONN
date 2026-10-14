import XCTest
@testable import PLONNCore

final class SocialAuthPolicyTests: XCTestCase {
    private let owner = UUID(), other = UUID()
    private let nonce = SocialAuthNonce.encode(Data(repeating: 42, count: 32))
    private func policy(bundle: String = "com.oakas.plonn", ready: SocialProviderReadiness = .init(apple: true, google: true), overrides: [String: Any] = [:]) -> SocialAuthPolicy {
        var flags: [String: Any] = ["PLONNAppleAuthEnabled": true, "PLONNAppleAuthReady": true,
                                  "PLONNGoogleAuthEnabled": true, "PLONNGoogleAuthReady": true, "PLONNIdentityLinkingReady": true]
        flags.merge(overrides) { _, new in new }
        return SocialAuthPolicy(bundleIdentifier: bundle, info: flags, readiness: ready)
    }
    private func appleToken(nonce: String) throws -> String {
        let payload = try JSONSerialization.data(withJSONObject: ["nonce": SocialAuthNonce.hash(nonce)])
        return "header." + SocialAuthNonce.encode(payload) + ".signature"
    }
    func testFlagsAndObservedProviderReadinessMustBothBeStrictlyTrue() throws {
        let disabled = SocialAuthPolicy(bundleIdentifier: "com.oakas.plonn", info: [:], readiness: .init(apple: true, google: true))
        XCTAssertFalse(disabled.needsReadinessCheck)
        XCTAssertFalse(disabled.available(.apple, action: .signIn)); XCTAssertFalse(disabled.available(.google, action: .signIn))
        let unobserved = policy(ready: .init())
        XCTAssertTrue(unobserved.needsReadinessCheck)
        XCTAssertFalse(unobserved.available(.apple, action: .signIn)); XCTAssertFalse(unobserved.available(.google, action: .signIn))
        for value in ["true", NSNumber(value: 1), false] as [Any] {
            XCTAssertFalse(policy(overrides: ["PLONNAppleAuthEnabled": value]).available(.apple, action: .signIn))
            XCTAssertFalse(policy(overrides: ["PLONNGoogleAuthReady": value]).available(.google, action: .signIn))
        }
        XCTAssertFalse(policy(bundle: "com.example.wrong").available(.apple, action: .signIn))
        XCTAssertFalse(policy(overrides: ["PLONNIdentityLinkingReady": false]).available(.google, action: .linkIdentity))
        XCTAssertTrue(policy(overrides: ["PLONNIdentityLinkingReady": false]).available(.google, action: .signIn))
        let actual = SocialProviderReadiness.decode(Data(#"{"external":{"apple":false,"google":false}}"#.utf8))
        XCTAssertEqual(actual, .init())
        XCTAssertEqual(SocialProviderReadiness.decode(Data(#"{"external":{"apple":1,"google":"true"}}"#.utf8)), .init())
        XCTAssertEqual(SocialProviderReadiness.decode(Data(repeating: 32, count: 16_385)), .init())
        XCTAssertEqual(SocialProviderReadiness.decode(Data(#"{"external":{"apple":true,"google":true}}"#.utf8)), .init(apple: true, google: true))
    }
    func testSignInAndLinkingRequireDifferentExplicitOwnerIntent() {
        var state = SocialAuthAttemptState()
        XCTAssertNil(state.begin(provider: .google, action: .linkIdentity, policy: policy()))
        let signIn = state.begin(provider: .google, action: .signIn, policy: policy())!
        XCTAssertNil(state.begin(provider: .apple, action: .signIn, policy: policy(), nonce: nonce))
        state.cancel(signIn); state.switchOwner(owner)
        XCTAssertNil(state.begin(provider: .google, action: .signIn, policy: policy()))
        let link = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        XCTAssertFalse(state.finish(link, currentOwner: owner, sessionOwner: other, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.isCurrent(link, currentOwner: owner))
        XCTAssertFalse(state.beginCommit(link, currentOwner: owner, sessionOwner: other, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.beginCommit(link, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.finish(link, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
    }
    func testCancellationAndOwnerRoundTripRejectLateOrRepeatedCallbacks() {
        var state = SocialAuthAttemptState(); state.switchOwner(owner)
        let old = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        state.cancel(old)
        XCTAssertFalse(state.finish(old, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        let replacement = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        state.cancel(old)
        XCTAssertTrue(state.isCurrent(replacement, currentOwner: owner))
        state.switchOwner(other); state.switchOwner(owner)
        XCTAssertFalse(state.isCurrent(replacement, currentOwner: owner))
        let current = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        XCTAssertTrue(state.beginCommit(current, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.finish(current, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertFalse(state.finish(current, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
    }
    func testVerifiedSessionMustBeUnexpiredAndNonanonymousBeforeAcceptance() {
        var state = SocialAuthAttemptState()
        let attempt = state.begin(provider: .google, action: .signIn, policy: policy())!
        XCTAssertFalse(state.finish(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: true, isExpired: false))
        XCTAssertFalse(state.finish(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: false, isExpired: true))
        XCTAssertFalse(state.finish(attempt, currentOwner: other, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertFalse(state.beginCommit(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: true, isExpired: false))
        XCTAssertFalse(state.beginCommit(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: false, isExpired: true))
        XCTAssertTrue(state.beginCommit(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.finish(attempt, currentOwner: nil, sessionOwner: owner, isAnonymous: false, isExpired: false))
    }
    func testVerifiedCommitDeniesCancellationAndOwnerChangesUntilPersistenceFinishes() {
        var state = SocialAuthAttemptState(); state.switchOwner(owner)
        let attempt = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        XCTAssertTrue(state.beginCommit(attempt, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.isCommitting)
        XCTAssertFalse(state.acceptsOAuthCallback(URL(string: "com.oakas.plonn://auth?code=late-code")!, attempt: attempt, currentOwner: owner, callback: policy().callback!))
        XCTAssertFalse(state.cancel(attempt))
        XCTAssertFalse(state.switchOwner(other)); XCTAssertFalse(state.switchOwner(nil))
        XCTAssertEqual(state.owner, owner)
        XCTAssertTrue(state.isCurrent(attempt, currentOwner: owner))
        XCTAssertNil(state.begin(provider: .google, action: .linkIdentity, policy: policy()))
        XCTAssertFalse(state.finish(attempt, currentOwner: owner, sessionOwner: other, isAnonymous: false, isExpired: false))
        XCTAssertTrue(state.finish(attempt, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        XCTAssertFalse(state.isCommitting)
        XCTAssertTrue(state.switchOwner(other))
        XCTAssertFalse(state.isCurrent(attempt, currentOwner: other))
    }
    func testFailedPersistenceReleasesCommitWithoutAdoptingOrRevivingOldCallback() {
        var state = SocialAuthAttemptState(); state.switchOwner(owner)
        let attempt = state.begin(provider: .apple, action: .linkIdentity, policy: policy(), nonce: nonce)!
        XCTAssertTrue(state.beginCommit(attempt, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
        state.endAfterPersistence(attempt)
        XCTAssertFalse(state.isCommitting); XCTAssertNil(state.pending); XCTAssertEqual(state.owner, owner)
        let next = state.begin(provider: .google, action: .linkIdentity, policy: policy())!
        state.endAfterPersistence(attempt)
        XCTAssertTrue(state.isCurrent(next, currentOwner: owner))
        XCTAssertFalse(state.finish(attempt, currentOwner: owner, sessionOwner: owner, isAnonymous: false, isExpired: false))
    }
    func testAppleNonceUsesSecureBytesAndSHA256BindingWithAttemptState() throws {
        let random = try SocialAuthNonce.generate(), second = try SocialAuthNonce.generate()
        XCTAssertEqual(random.utf8.count, 43); XCTAssertNotEqual(random, second)
        XCTAssertEqual(SocialAuthNonce.hash("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        var state = SocialAuthAttemptState()
        XCTAssertNil(state.begin(provider: .apple, action: .signIn, policy: policy(), nonce: "short"))
        let attempt = state.begin(provider: .apple, action: .signIn, policy: policy(), nonce: nonce)!
        let token = try appleToken(nonce: nonce)
        XCTAssertTrue(state.acceptsAppleCredential(token: token, state: attempt.id.uuidString, attempt: attempt, currentOwner: nil))
        XCTAssertFalse(state.acceptsAppleCredential(token: token, state: nil, attempt: attempt, currentOwner: nil))
        XCTAssertFalse(state.acceptsAppleCredential(token: token, state: UUID().uuidString, attempt: attempt, currentOwner: nil))
        XCTAssertFalse(state.acceptsAppleCredential(token: try appleToken(nonce: random), state: attempt.id.uuidString, attempt: attempt, currentOwner: nil))
        XCTAssertFalse(state.acceptsAppleCredential(token: "not-a-token", state: attempt.id.uuidString, attempt: attempt, currentOwner: nil))
        state.cancel(attempt)
        XCTAssertFalse(state.acceptsAppleCredential(token: token, state: attempt.id.uuidString, attempt: attempt, currentOwner: nil))
    }
    func testGoogleCallbacksAreBoundToAttemptAppAndPKCECodeShape() throws {
        var state = SocialAuthAttemptState()
        let attempt = state.begin(provider: .google, action: .signIn, policy: policy())!
        let callback = policy().callback!
        let accepted = URL(string: "com.oakas.plonn://auth?code=one-time-code")!
        XCTAssertTrue(state.acceptsOAuthCallback(accepted, attempt: attempt, currentOwner: nil, callback: callback))
        for value in ["com.plonn.preview://auth?code=x", "com.oakas.plonn://auth/other?code=x", "com.oakas.plonn://auth?code=", "com.oakas.plonn://auth?code=x&code=y", "com.oakas.plonn://auth?code=x&error=denied", "com.oakas.plonn://auth#access_token=token", "com.oakas.plonn://auth?code=x#type=recovery"] {
            XCTAssertFalse(state.acceptsOAuthCallback(try XCTUnwrap(URL(string: value)), attempt: attempt, currentOwner: nil, callback: callback), value)
        }
        state.cancel(attempt)
        XCTAssertFalse(state.acceptsOAuthCallback(accepted, attempt: attempt, currentOwner: nil, callback: callback))
    }
    func testGoogleAuthorizationOpensOnlySupabaseOrGoogleAuthOrigins() {
        let origin = URL(string: "https://iwjscqdsebszqnrhbhrl.supabase.co")!
        for value in [origin.absoluteString + "/auth/v1/authorize?provider=google", "https://accounts.google.com/o/oauth2/v2/auth?client_id=public"] {
            XCTAssertTrue(SocialAuthPolicy.allowedGoogleAuthorizationURL(URL(string: value)!, authURL: origin))
        }
        for value in ["http://accounts.google.com/o/oauth2/v2/auth", "https://accounts.google.com.evil.test/o/oauth2/v2/auth", "https://accounts.google.com:443/o/oauth2/v2/auth", "https://user:secret@accounts.google.com/o/oauth2/v2/auth", "https://example.com/auth/v1/authorize", origin.absoluteString + "/other"] {
            XCTAssertFalse(SocialAuthPolicy.allowedGoogleAuthorizationURL(URL(string: value)!, authURL: origin))
        }
    }
}
