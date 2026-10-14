import Foundation
import CoreFoundation
import CryptoKit
import Security

enum SocialAuthProvider: String, CaseIterable, Sendable { case apple, google }
enum SocialAuthAction: Equatable, Sendable { case signIn, linkIdentity }
enum SocialAuthError: Error { case unavailable, invalidAttempt, invalidCallback, invalidCredential, randomFailure, noPresentationAnchor }

/// Public provider settings can disable a configured build; they cannot enable it.
struct SocialProviderReadiness: Equatable, Sendable {
    var apple = false
    var google = false
    static func decode(_ data: Data) -> SocialProviderReadiness {
        guard data.count <= 16_384,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let external = object["external"] as? [String: Any] else { return .init() }
        func enabled(_ key: String) -> Bool {
            guard let value = external[key] as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return false }
            return value.boolValue
        }
        return .init(apple: enabled("apple"), google: enabled("google"))
    }
}

struct SocialAuthPolicy: Sendable {
    let callback: AuthCallbackPolicy?
    let appleConfigured: Bool
    let googleConfigured: Bool
    let linkingConfigured: Bool
    let readiness: SocialProviderReadiness

    init(bundleIdentifier: String?, info: [String: Any], readiness: SocialProviderReadiness = .init()) {
        callback = AuthCallbackPolicy(bundleIdentifier: bundleIdentifier)
        func flag(_ key: String) -> Bool {
            guard let value = info[key] as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return false }
            return value.boolValue
        }
        appleConfigured = flag("PLONNAppleAuthEnabled") && flag("PLONNAppleAuthReady")
        googleConfigured = flag("PLONNGoogleAuthEnabled") && flag("PLONNGoogleAuthReady")
        linkingConfigured = flag("PLONNIdentityLinkingReady")
        self.readiness = readiness
    }

    var needsReadinessCheck: Bool { callback != nil && (appleConfigured || googleConfigured) }
    func available(_ provider: SocialAuthProvider, action: SocialAuthAction) -> Bool {
        guard callback != nil, action != .linkIdentity || linkingConfigured else { return false }
        return provider == .apple ? appleConfigured && readiness.apple : googleConfigured && readiness.google
    }
    static func allowedGoogleAuthorizationURL(_ url: URL, authURL: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.scheme == "https", components.user == nil, components.password == nil,
              components.port == nil, components.fragment == nil else { return false }
        if components.host == authURL.host {
            return ["/auth/v1/authorize", "/auth/v1/user/identities/authorize"].contains(components.path)
        }
        return components.host == "accounts.google.com" && ["/o/oauth2/v2/auth", "/o/oauth2/auth"].contains(components.path)
    }
}

enum SocialAuthNonce {
    static func generate() throws -> String {
        var bytes = Data(count: 32)
        let status = bytes.withUnsafeMutableBytes { buffer in
            SecRandomCopyBytes(kSecRandomDefault, buffer.count, buffer.baseAddress!)
        }
        guard status == errSecSuccess else { throw SocialAuthError.randomFailure }
        return encode(bytes)
    }
    static func encode(_ bytes: Data) -> String {
        bytes.base64EncodedString().replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
    }
    static func hash(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    }
    /// Checks nonce binding only. Supabase must still verify signature/audience.
    static func tokenMatches(_ token: String, nonce: String) -> Bool {
        guard token.utf8.count <= 8192 else { return false }
        let parts = token.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty }) else { return false }
        var payload = String(parts[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        payload += String(repeating: "=", count: (4 - payload.count % 4) % 4)
        guard let bytes = Data(base64Encoded: payload),
              let object = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] else { return false }
        return object["nonce"] as? String == hash(nonce)
    }
}

struct SocialAuthAttempt: Equatable, Sendable {
    let id: UUID
    let owner: UUID?
    let provider: SocialAuthProvider
    let action: SocialAuthAction
    let nonce: String?
}

/// Local sequencing, not identity verification. No email matching or merging.
struct SocialAuthAttemptState: Sendable {
    private(set) var owner: UUID?
    private(set) var pending: SocialAuthAttempt?
    private(set) var committedOwner: UUID?
    var isCommitting: Bool { committedOwner != nil }

    @discardableResult mutating func switchOwner(_ next: UUID?) -> Bool {
        guard !isCommitting else { return false }
        guard owner != next else { return true }
        owner = next; pending = nil
        return true
    }
    mutating func begin(provider: SocialAuthProvider, action: SocialAuthAction, policy: SocialAuthPolicy, nonce: String? = nil) -> SocialAuthAttempt? {
        guard pending == nil, policy.available(provider, action: action),
              action == .signIn ? owner == nil : owner != nil,
              provider != .apple || (nonce?.utf8.count == 43 && nonce?.range(of: #"^[A-Za-z0-9_-]+$"#, options: .regularExpression) != nil) else { return nil }
        let attempt = SocialAuthAttempt(id: UUID(), owner: owner, provider: provider, action: action, nonce: provider == .apple ? nonce : nil)
        pending = attempt
        return attempt
    }
    func isCurrent(_ attempt: SocialAuthAttempt, currentOwner: UUID?) -> Bool {
        pending == attempt && owner == attempt.owner && currentOwner == attempt.owner
    }
    func acceptsOAuthCallback(_ url: URL, attempt: SocialAuthAttempt, currentOwner: UUID?, callback: AuthCallbackPolicy) -> Bool {
        guard !isCommitting, attempt.provider == .google, isCurrent(attempt, currentOwner: currentOwner), callback.accepts(url),
              let components = URLComponents(url: url, resolvingAgainstBaseURL: false), components.fragment == nil else { return false }
        let codes = (components.queryItems ?? []).filter { $0.name == "code" }
        return codes.count == 1 && codes[0].value?.isEmpty == false && (codes[0].value?.utf8.count ?? 0) <= 2048 &&
            !(components.queryItems ?? []).contains(where: { ["error", "error_description", "access_token", "refresh_token"].contains($0.name) })
    }
    func acceptsAppleCredential(token: String, state: String?, attempt: SocialAuthAttempt, currentOwner: UUID?) -> Bool {
        guard !isCommitting, attempt.provider == .apple, isCurrent(attempt, currentOwner: currentOwner), state == attempt.id.uuidString,
              let nonce = attempt.nonce else { return false }
        return SocialAuthNonce.tokenMatches(token, nonce: nonce)
    }
    mutating func beginCommit(_ attempt: SocialAuthAttempt, currentOwner: UUID?, sessionOwner: UUID, isAnonymous: Bool, isExpired: Bool) -> Bool {
        guard isCurrent(attempt, currentOwner: currentOwner), !isAnonymous, !isExpired,
              !isCommitting, attempt.action != .linkIdentity || sessionOwner == attempt.owner else { return false }
        committedOwner = sessionOwner
        return true
    }
    mutating func finish(_ attempt: SocialAuthAttempt, currentOwner: UUID?, sessionOwner: UUID, isAnonymous: Bool, isExpired: Bool) -> Bool {
        guard isCurrent(attempt, currentOwner: currentOwner), isCommitting, committedOwner == sessionOwner,
              !isAnonymous, !isExpired else { return false }
        pending = nil; committedOwner = nil
        return true
    }
    @discardableResult mutating func cancel(_ attempt: SocialAuthAttempt) -> Bool {
        guard pending?.id == attempt.id, !isCommitting else { return false }
        pending = nil
        return true
    }
    /// Only the task that completed/failed SDK persistence may release commit.
    mutating func endAfterPersistence(_ attempt: SocialAuthAttempt) {
        guard pending?.id == attempt.id else { return }
        pending = nil; committedOwner = nil
    }
}
