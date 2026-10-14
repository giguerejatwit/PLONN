import XCTest
@testable import PLONNCore

final class PLONNServicePolicyTests: XCTestCase {
    func testDefaultMissingMalformedAndPartialConfigurationFailClosed() {
        for info: [String: Any] in [[:], ["PLONNChatEnabled": true], ["PLONNPublicPagesApproved": true],
                                  ["PLONNChatEnabled": "true", "PLONNPublicPagesApproved": true],
                                  ["PLONNChatEnabled": 1, "PLONNPublicPagesApproved": true]] {
            XCTAssertFalse(PLONNServicePolicy(info: info).chatAvailable)
        }
        XCTAssertFalse(PLONNServicePolicy().publicPagesApproved)
    }
    func testPermissionIsBoundToSignedInOwnerAndApprovedPages() {
        let owner = UUID(), other = UUID()
        let policy = PLONNServicePolicy(info: ["PLONNChatEnabled": true, "PLONNPublicPagesApproved": true])
        XCTAssertTrue(policy.mayUseChat(ownerID: owner, consentOwnerID: owner))
        XCTAssertFalse(policy.mayUseChat(ownerID: other, consentOwnerID: owner))
        XCTAssertFalse(policy.mayUseChat(ownerID: nil, consentOwnerID: nil))
        XCTAssertFalse(policy.mayUseChat(ownerID: owner, consentOwnerID: nil))
        XCTAssertFalse(PLONNServicePolicy(chatEnabled: true).mayUseChat(ownerID: owner, consentOwnerID: owner))
        XCTAssertEqual(PLONNServicePolicy.privacyURL.absoluteString, "https://oakas.ai/plonn/privacy")
        XCTAssertEqual(PLONNServicePolicy.supportURL.absoluteString, "https://oakas.ai/plonn/support")
    }
}
