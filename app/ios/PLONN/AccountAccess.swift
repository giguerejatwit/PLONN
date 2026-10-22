import Foundation
import Combine

enum AccountFeature: String, CaseIterable, Sendable {
    case odds, aiResearch, savedResearch, privateModels
    var title: String {
        switch self {
        case .odds: return "odds"
        case .aiResearch: return "AI research"
        case .savedResearch: return "saved research"
        case .privateModels: return "private models"
        }
    }
    var detail: String {
        switch self {
        case .odds: return "Sign in to view sportsbook lines and odds. Games and statistics are available in preview."
        case .aiResearch: return "Sign in to ask PLONN questions and explore AI matchup research."
        case .savedResearch: return "Sign in to save selections and return to your research."
        case .privateModels: return "Sign in to build models and access your private models and results."
        }
    }
}

enum AccountAccessPolicy {
    static func permits(ownerID: UUID?, isAnonymous: Bool?) -> Bool {
        ownerID != nil && isAnonymous == false
    }
}

struct AccountSignInRequest: Identifiable, Equatable {
    let id = UUID()
    var feature: AccountFeature?
    var creating = false
}

/// A cancelled prompt must never replay an earlier save or selection on a later login.
@MainActor final class AccountAccessCoordinator: ObservableObject {
    @Published private(set) var request: AccountSignInRequest?
    private var continuation: (() -> Void)?

    func present(feature: AccountFeature? = nil, creating: Bool = false) {
        continuation = nil
        request = AccountSignInRequest(feature: feature, creating: creating)
    }
    func require(_ feature: AccountFeature, ownerID: UUID?, isAnonymous: Bool?, then action: @escaping () -> Void) {
        if AccountAccessPolicy.permits(ownerID: ownerID, isAnonymous: isAnonymous) { action(); return }
        present(feature: feature)
        continuation = action
    }
    func complete(ownerID: UUID?, isAnonymous: Bool?) {
        guard AccountAccessPolicy.permits(ownerID: ownerID, isAnonymous: isAnonymous) else { return }
        let action = continuation
        continuation = nil; request = nil
        action?()
    }
    func cancel() { continuation = nil; request = nil }
}
