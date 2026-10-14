import Foundation

struct VerifiedProductEntitlement: Equatable {
    let productID: String
    let accountToken: UUID?
    let expiration: Date?
    let revoked: Bool
    func grantsAccess(to owner: UUID?, configuredProducts: Set<String>, now: Date = Date()) -> Bool {
        guard let owner, accountToken == owner, configuredProducts.contains(productID), !revoked else { return false }
        return expiration.map { $0 > now } ?? true
    }
}
