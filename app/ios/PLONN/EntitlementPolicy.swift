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

/// The server checks Apple's latest status. StoreKit alone cannot grant cloud access.
struct VerifiedUsageAllowance: Decodable, Equatable {
    let version: String
    let aiRequestsPerDay: Int
    let modelRunsPerDay: Int
    let resetTimeZone: String
    let rollover: Bool
    let counting: String
    var isSupported: Bool {
        version == "plonn_pro_daily_v1" && aiRequestsPerDay == 20 && modelRunsPerDay == 5
            && resetTimeZone == "UTC" && !rollover && counting == "started_attempts"
    }
}

struct VerifiedPlanStatus: Decodable, Equatable {
    let hasPro: Bool
    let betaAccess: Bool
    let checkoutReady: Bool
    let productID: String?
    let expiresAt: String?
    let checkoutMode: String?
    let environment: String?
    let usageAllowance: VerifiedUsageAllowance?
    init(hasPro: Bool, betaAccess: Bool, checkoutReady: Bool, productID: String?, expiresAt: String?,
         checkoutMode: String? = nil, environment: String? = nil, usageAllowance: VerifiedUsageAllowance? = nil) {
        self.hasPro = hasPro; self.betaAccess = betaAccess; self.checkoutReady = checkoutReady
        self.productID = productID; self.expiresAt = expiresAt
        self.checkoutMode = checkoutMode; self.environment = environment
        self.usageAllowance = usageAllowance
    }
    func grantsProAccess(now: Date = Date()) -> Bool {
        hasPro && (expiration ?? .distantPast) > now
    }
    var expiration: Date? {
        guard let expiresAt else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: expiresAt) ?? ISO8601DateFormatter().date(from: expiresAt)
    }
}

enum SubscriptionCheckoutPolicy {
    /// Call only after StoreKit verifies the AppTransaction signature.
    /// Apple omits appID in Sandbox; production must match the App Store app ID.
    static func appleEnvironment(bundleID: String, appID: UInt64?, environment: String) -> String? {
        guard bundleID == "com.oakas.plonn" else { return nil }
        switch environment {
        case "Sandbox": return appID == nil || appID == 6821990305 ? environment : nil
        case "Production": return appID == 6821990305 ? environment : nil
        default: return nil
        }
    }

    static func permits(plan: VerifiedPlanStatus?, owner: UUID?, salesEnabled: Bool, appleEnvironment: String?) -> Bool {
        guard owner != nil, let plan, plan.checkoutReady else { return false }
        switch plan.checkoutMode {
        case "sandbox": return appleEnvironment == "Sandbox"
        case "production": return salesEnabled && appleEnvironment == "Production" && plan.usageAllowance?.isSupported == true
        default: return false
        }
    }
}
