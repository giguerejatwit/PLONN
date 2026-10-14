import Foundation

/// The server is authoritative. This state only controls account-scoped presentation.
struct ReferralReply: Decodable {
    let owner_id: UUID
    let status: String
    let code: String?
}

struct ReferralRequest: Sendable, Equatable {
    let owner: UUID
    let token: UUID
    /// Nil means an explicitly requested status refresh, never a blank code.
    let code: String?
}

struct ReferralState {
    private(set) var owner: UUID?
    private(set) var request: UUID?
    private(set) var appliedCode: String?
    private(set) var message = "Add a referral code if someone referred you."
    private var refreshingStatus = false
    private var automaticallyAttemptedCodes = Set<String>()
    var busy: Bool { request != nil }

    static func normalized(_ input: String) -> String {
        input.trimmingCharacters(in: .whitespacesAndNewlines).uppercased(with: Locale(identifier: "en_US_POSIX"))
    }

    static func pendingCode(_ input: String?) -> String? {
        guard let input else { return nil }
        let code = normalized(input)
        return code.isEmpty ? nil : code
    }

    mutating func switchOwner(_ next: UUID?) {
        guard owner != next else { return }
        self = ReferralState()
        owner = next
    }

    mutating func begin(owner expected: UUID, code: String? = nil) -> UUID? {
        guard owner == expected, request == nil,
              code == nil || Self.pendingCode(code) != nil else { return nil }
        let token = UUID()
        request = token
        refreshingStatus = code == nil
        return token
    }

    mutating func beginRequest(owner expected: UUID, code: String? = nil) -> ReferralRequest? {
        guard let token = begin(owner: expected, code: code) else { return nil }
        return ReferralRequest(owner: expected, token: token, code: Self.pendingCode(code))
    }

    /// Reserve the owner/token before scheduling asynchronous work. Repeated
    /// auth callbacks cannot replay an already attempted pending code.
    mutating func beginAutomatic(owner expected: UUID, pendingCode: String?) -> ReferralRequest? {
        guard let code = Self.pendingCode(pendingCode), appliedCode == nil,
              !automaticallyAttemptedCodes.contains(code),
              let operation = beginRequest(owner: expected, code: code) else { return nil }
        automaticallyAttemptedCodes.insert(code)
        return operation
    }

    func isCurrent(_ operation: ReferralRequest) -> Bool {
        owner == operation.owner && request == operation.token
    }

    mutating func finish(_ reply: ReferralReply, token: UUID) {
        guard request == token else { return }
        guard owner == reply.owner_id else { fail(token: token); return }
        request = nil
        switch reply.status {
        case "applied", "already_applied":
            guard let code = Self.pendingCode(reply.code) else { message = confirmationFailure; return }
            appliedCode = code
            message = "Referral code applied: \(code)."
        case "already_attributed":
            guard let code = Self.pendingCode(reply.code) else { message = confirmationFailure; return }
            appliedCode = code
            message = "Your account already has a referral code: \(code)."
        case "invalid": message = "That code isn’t valid or has expired. Check it and try again."
        case "needs_verification": message = "Confirm your email, then sign in to apply your code."
        case "none": appliedCode = nil; message = "Add a referral code if someone referred you."
        default: message = confirmationFailure
        }
    }

    mutating func fail(token: UUID) {
        guard request == token else { return }
        request = nil
        message = refreshingStatus ? "Couldn’t refresh your referral status. Try again shortly." : "Couldn’t check your code. Try again shortly."
    }

    private var confirmationFailure: String {
        refreshingStatus ? "Couldn’t confirm your referral status. Try refreshing again." : "Couldn’t confirm your code. Try again."
    }
}
