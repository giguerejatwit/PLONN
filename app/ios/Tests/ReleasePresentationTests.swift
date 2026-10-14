import XCTest
@testable import PLONNCore

final class ReleasePresentationTests: XCTestCase {
    func testArbitraryServiceDiagnosticsCannotReachCustomerCopy() {
        let diagnostic = "Traceback /data/private/customer/run-id token=secret"
        for body: [String: Any] in [
            ["message": diagnostic],
            ["status": "ok", "message": diagnostic],
            ["reason": diagnostic, "inference": ["reason": diagnostic]],
            ["inference": ["status": "ok"], "message": diagnostic]
        ] {
            let message = LabServiceContract.userFacingMessage(body)
            XCTAssertFalse(message.contains(diagnostic))
            XCTAssertFalse(message.contains("/data"))
            XCTAssertFalse(message.contains("secret"))
            XCTAssertFalse(message.isEmpty)
        }
    }
    func testKnownFailuresStillExplainHowToRecover() {
        XCTAssertTrue(LabServiceContract.userFacingMessage(["reason": "authentication_required"]).contains("Sign in"))
        XCTAssertTrue(LabServiceContract.userFacingMessage(["inference": ["reason": "stale_target"]]).contains("Refresh"))
        XCTAssertTrue(LabServiceContract.userFacingMessage(["training": ["reason": "insufficient_training_rows"]]).contains("historical data"))
        XCTAssertTrue(LabServiceContract.userFacingMessage(["reason": "training_interrupted"]).contains("Retrain"))
        XCTAssertTrue(LabServiceContract.userFacingMessage(["reason": "training_quota_exceeded"]).contains("limit"))
    }
}
