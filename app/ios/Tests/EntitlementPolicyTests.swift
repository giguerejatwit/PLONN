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
}
