import XCTest
@testable import PLONNCore

final class EntitlementPolicyTests: XCTestCase {
    let owner=UUID()
    let now=Date(timeIntervalSince1970:1_000_000)
    // Test-only identifiers, never shipped as StoreKit configuration.
    let configured:Set<String>=["test.fixture.product"]
    func testOnlyConfiguredOwnerBoundUnexpiredEntitlementGrantsAccess() {
        let valid=VerifiedProductEntitlement(productID:"test.fixture.product",accountToken:owner,expiration:now.addingTimeInterval(60),revoked:false)
        XCTAssertTrue(valid.grantsAccess(to:owner,configuredProducts:configured,now:now))
        XCTAssertFalse(valid.grantsAccess(to:nil,configuredProducts:configured,now:now))
        XCTAssertFalse(valid.grantsAccess(to:UUID(),configuredProducts:configured,now:now))
        XCTAssertFalse(valid.grantsAccess(to:owner,configuredProducts:[],now:now))
        XCTAssertFalse(valid.grantsAccess(to:owner,configuredProducts:configured,now:now.addingTimeInterval(60)))
    }
    func testRevocationAndUnboundLegacyPurchaseFailClosed() {
        let revoked=VerifiedProductEntitlement(productID:"test.fixture.product",accountToken:owner,expiration:nil,revoked:true)
        XCTAssertFalse(revoked.grantsAccess(to:owner,configuredProducts:configured,now:now))
        let unbound=VerifiedProductEntitlement(productID:"test.fixture.product",accountToken:nil,expiration:nil,revoked:false)
        XCTAssertFalse(unbound.grantsAccess(to:owner,configuredProducts:configured,now:now))
    }
    func testServerPlanRequiresProAndFutureExpiry() throws {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let valid = VerifiedPlanStatus(hasPro: true, betaAccess: false, checkoutReady: true, productID: nil, expiresAt: "2099-01-01T00:00:00.000Z")
        XCTAssertTrue(valid.grantsProAccess(now: now))
        XCTAssertFalse(VerifiedPlanStatus(hasPro: true, betaAccess: false, checkoutReady: true, productID: nil, expiresAt: nil).grantsProAccess(now: now))
        XCTAssertFalse(VerifiedPlanStatus(hasPro: false, betaAccess: true, checkoutReady: true, productID: nil, expiresAt: "2099-01-01T00:00:00Z").grantsProAccess(now: now))
    }
    private func plan(mode: String?, ready: Bool = true) -> VerifiedPlanStatus {
        VerifiedPlanStatus(hasPro: false, betaAccess: false, checkoutReady: ready,
            productID: nil, expiresAt: nil, checkoutMode: mode,
            usageAllowance: VerifiedUsageAllowance(version: "plonn_pro_daily_v1", aiRequestsPerDay: 20,
                modelRunsPerDay: 5, resetTimeZone: "UTC", rollover: false, counting: "started_attempts"))
    }
    func testProductionCheckoutRequiresSupportedServerAllowance() throws {
        let missing = VerifiedPlanStatus(hasPro: false, betaAccess: false, checkoutReady: true,
            productID: nil, expiresAt: nil, checkoutMode: "production")
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: missing, owner: owner, salesEnabled: true, appleEnvironment: "Production"))
        let contract = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("backend/supabase/functions/_shared/pro_allowance.json")
        let data = try Data(contentsOf: contract)
        XCTAssertTrue(try JSONDecoder().decode(VerifiedUsageAllowance.self, from: data).isSupported)
        for altered in [String(decoding: data, as: UTF8.self).replacingOccurrences(of: "20", with: "200"),
                        String(decoding: data, as: UTF8.self).replacingOccurrences(of: "UTC", with: "local")] {
            XCTAssertFalse(try JSONDecoder().decode(VerifiedUsageAllowance.self, from: Data(altered.utf8)).isSupported)
        }
    }
    func testSandboxCheckoutRequiresServerAuthorizationAndVerifiedAppleSandbox() {
        XCTAssertTrue(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox"), owner: owner, salesEnabled: false, appleEnvironment: "Sandbox"))
        for environment in [nil, "Production", "Xcode", "unknown"] as [String?] {
            XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox"), owner: owner, salesEnabled: true, appleEnvironment: environment))
        }
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox", ready: false), owner: owner, salesEnabled: false, appleEnvironment: "Sandbox"))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox"), owner: nil, salesEnabled: false, appleEnvironment: "Sandbox"))
    }
    func testSandboxAppTransactionWithApplesDocumentedNilAppIDCanReachAuthorizedCheckout() {
        let environment = SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.oakas.plonn", appID: nil, environment: "Sandbox")
        XCTAssertEqual(environment, "Sandbox")
        XCTAssertTrue(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox"), owner: owner,
            salesEnabled: false, appleEnvironment: environment))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "sandbox", ready: false), owner: owner,
            salesEnabled: false, appleEnvironment: environment))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "production"), owner: owner,
            salesEnabled: true, appleEnvironment: environment))
    }
    func testProductionAppIdentityStillRequiresTheExactAppID() {
        XCTAssertEqual(SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.oakas.plonn", appID: 6821990305, environment: "Production"), "Production")
        for appID in [nil, 0, 12345] as [UInt64?] {
            XCTAssertNil(SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.oakas.plonn", appID: appID, environment: "Production"))
        }
    }
    func testWrongBundleUnexpectedSandboxIDAndLocalEnvironmentsCannotEnableCheckout() {
        for environment in ["Sandbox", "Production"] {
            XCTAssertNil(SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.other.app", appID: 6821990305, environment: environment))
        }
        XCTAssertNil(SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.oakas.plonn", appID: 12345, environment: "Sandbox"))
        for environment in ["Xcode", "unknown", "", "sandbox"] {
            XCTAssertNil(SubscriptionCheckoutPolicy.appleEnvironment(bundleID: "com.oakas.plonn", appID: nil, environment: environment))
        }
    }
    func testProductionCheckoutRemainsDisabledUntilReleaseActivation() {
        XCTAssertTrue(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "production"), owner: owner, salesEnabled: true, appleEnvironment: "Production"))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "production"), owner: owner, salesEnabled: false, appleEnvironment: "Production"))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "production"), owner: owner, salesEnabled: true, appleEnvironment: "Sandbox"))
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: "production", ready: false), owner: owner, salesEnabled: true, appleEnvironment: "Production"))
    }
    func testMissingOrUnknownCheckoutModeFailsClosed() {
        for mode in [nil, "unavailable", "unknown"] as [String?] {
            XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: plan(mode: mode), owner: owner, salesEnabled: true, appleEnvironment: "Production"))
        }
        XCTAssertFalse(SubscriptionCheckoutPolicy.permits(plan: nil, owner: owner, salesEnabled: true, appleEnvironment: "Production"))
    }
    func testLegacyAndSandboxServerResponsesDecodeWithoutInventingPaidAccess() throws {
        let legacy = Data(#"{"hasPro":false,"betaAccess":true,"checkoutReady":false,"productID":null,"expiresAt":null}"#.utf8)
        let old = try JSONDecoder().decode(VerifiedPlanStatus.self, from: legacy)
        XCTAssertNil(old.checkoutMode)
        XCTAssertNil(old.environment)
        XCTAssertFalse(old.grantsProAccess())
        let sandbox = Data(#"{"hasPro":true,"betaAccess":false,"checkoutReady":true,"productID":"com.oakas.plonn.pro.monthly","expiresAt":"2099-01-01T00:00:00Z","checkoutMode":"sandbox","environment":"Sandbox"}"#.utf8)
        let test = try JSONDecoder().decode(VerifiedPlanStatus.self, from: sandbox)
        XCTAssertTrue(test.grantsProAccess(now: now))
        XCTAssertEqual(test.environment, "Sandbox")
        XCTAssertTrue(SubscriptionCheckoutPolicy.permits(plan: test, owner: owner, salesEnabled: false, appleEnvironment: "Sandbox"))
    }
    func testDeletionClearsOnlySelectedAccountsSubmissionRecords() throws {
        let suite = "deletion-test-" + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = LabSubmissionStore(defaults: defaults)
        let other = UUID(), payload = Data("test".utf8)
        _ = try store.prepare(owner: owner, payload: payload)
        _ = try store.prepare(owner: other, payload: payload)
        store.deleteAccountRecords(owner: owner)
        XCTAssertNil(store.record(owner: owner, payload: payload))
        XCTAssertNotNil(store.record(owner: other, payload: payload))
    }
}
