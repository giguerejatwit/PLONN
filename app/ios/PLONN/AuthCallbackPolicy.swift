import Foundation

enum AccountAIConsentPolicy {
    static let version = "v2_ollama_cloud"
    static let metadataKey = "plonn_ai_consent"
    static let disclosure = "When you request AI, your question, recent conversation, and selected research are sent through PLONN to Ollama’s hosted Gemma 4 31B service. Ollama processes these prompts and responses and states that it does not log them or use them to train models. Player and model attachments are optional. Avoid sensitive personal information. AI runs only when you request it; you can use research without AI."

    static func preferenceKey(owner: UUID) -> String {
        "plonn.aiConsent.\(version).\(owner.uuidString.lowercased())"
    }

    static func permits(isAnonymous: Bool, metadataVersion: String?, localChoice: Bool?) -> Bool {
        guard !isAnonymous else { return false }
        return localChoice ?? (metadataVersion == version)
    }

    static func synchronizedChoice(requested: Bool, pending: Bool?, metadataVersion: String?) -> Bool {
        pending == requested && metadataVersion == (requested ? version : "declined_" + version)
    }

    static func mayBeginConsentSync(accountBusy: Bool, socialPending: Bool, consentUpdating: Bool) -> Bool {
        !accountBusy && !socialPending && !consentUpdating
    }
}

/// Records only callback structure; never retains or returns authentication values.
enum AuthCallbackPresence {
    static func category(for url: URL) -> String {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return "callback_invalid"
        }
        let query = components.queryItems ?? []
        let fragment = components.fragment.flatMap {
            URLComponents(string: "https://callback.invalid/?" + $0)?.queryItems
        } ?? []
        let items = query + fragment
        if items.contains(where: { ["error", "error_description", "error_code"].contains($0.name) }) {
            return "callback_error"
        }
        let codes = items.filter { $0.name == "code" }
        guard !codes.isEmpty else { return "callback_code_missing" }
        guard codes.count == 1 else { return "callback_code_ambiguous" }
        return codes[0].value?.isEmpty == false ? "callback_code_present" : "callback_code_empty"
    }
}

/// Reject incomplete or ambiguous email callbacks before invoking the SDK.
/// Payload values stay private; only bounded failure categories reach the UI.
enum EmailAuthCallbackValidation {
    static func failure(for url: URL, verifierPresent: Bool) -> String? {
        switch AuthCallbackPresence.category(for: url) {
        case "callback_error": return nil // Let the SDK verify and classify provider errors.
        case "callback_code_present": return verifierPresent ? nil : "sign_in_context"
        default: return "link_incomplete"
        }
    }
}

enum EmailAuthLinkError: Error {
    case incomplete, missingRequest, storageUnavailable
    var category: String {
        switch self {
        case .incomplete: return "link_incomplete"
        case .missingRequest: return "sign_in_context"
        case .storageUnavailable: return "auth_storage"
        }
    }
}

enum AuthFailurePresentation {
    static func category(serverCode: String?, httpStatus: Int?) -> String {
        switch serverCode {
        case "otp_expired", "flow_state_expired": return "link_expired"
        case "flow_state_not_found", "bad_code_verifier": return "recovery_context"
        case "over_email_send_rate_limit", "over_request_rate_limit": return "rate_limit"
        case "invalid_credentials": return "credentials"
        default:
            if httpStatus == 429 { return "rate_limit" }
            if let httpStatus, httpStatus >= 500 { return "auth_service" }
            return "auth_request"
        }
    }

    static func message(forCategory category: String) -> String {
        switch category {
        case "link_expired":
            return "This account link has expired or was already used. Request a new link and open it once on this device."
        case "recovery_context":
            return "This account link does not match the request on this device. Request a new link here and open it in the same app on this device."
        case "sign_in_context":
            return "This link doesn’t match a sign-in request in this app. Request a new link here, then open the newest email on this device."
        case "link_incomplete":
            return "This sign-in link is incomplete. Request a new link and open it directly from the newest email."
        case "auth_storage":
            return "PLONN couldn’t save your sign-in request securely. Unlock your device, reopen PLONN, and request a new link."
        case "network":
            return "Couldn’t reach the account service. Check your connection and try again."
        case "rate_limit":
            return "Too many account requests. Wait before requesting another link or trying again."
        case "credentials":
            return "Sign-in details could not be verified. Check your email and password."
        case "auth_service":
            return "The account service is temporarily unavailable. Try again later."
        default:
            return "The account request could not be completed. Check your connection and account details before trying again."
        }
    }
}

/// A short-lived local intent, evaluated only after the SDK verifies a session.
/// Callers supply fingerprints; raw email addresses and credentials are not stored here.
struct PasswordRecoveryIntent: Codable, Equatable, Sendable {
    let emailFingerprint: String
    let requestedAt: Date

    func permitsRecovery(verifiedEmailFingerprint: String?, now: Date) -> Bool {
        let elapsed = now.timeIntervalSince(requestedAt)
        return !emailFingerprint.isEmpty
            && verifiedEmailFingerprint == emailFingerprint
            && elapsed.isFinite && elapsed >= 0 && elapsed <= 3_600
    }
}

/// Keeps production and preview email links bound to their own app identity.
struct AuthCallbackPolicy: Equatable, Sendable {
    let scheme: String

    init?(bundleIdentifier: String?) {
        guard let bundleIdentifier,
              ["com.oakas.plonn", "com.plonn.preview"].contains(bundleIdentifier) else { return nil }
        scheme = bundleIdentifier
    }

    var redirectURL: URL { URL(string: "\(scheme)://auth")! }

    func accepts(_ url: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return false }
        return components.scheme?.lowercased() == scheme
            && components.host?.lowercased() == "auth"
            && components.user == nil && components.password == nil
            && components.port == nil && components.path.isEmpty
    }
}
